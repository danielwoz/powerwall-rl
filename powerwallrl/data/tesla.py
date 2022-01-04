""" This module provides a class for collecting tesla powerwall api data for a
given username.

For multiple reasons this probably won't work for complex setups like
multi-battery single site, or multi-site Tesla powerwall setups.
"""

# Author: Daniel Williams

__version__ = '0.0.1'

import logging
from teslapy import Tesla
from dateutil.parser import parse
import dateutil
import dateutil.tz
from datetime import datetime
from datetime import timedelta
import sqlite3
from ratelimit import limits, sleep_and_retry
from babel.dates import format_datetime

logger = logging.getLogger(__name__)


class TeslaPowerwallData(object):

  def __init__(self, username, database, local_timezone='Etc/UTC'):
    self.con = database
    self.username = username
    self.tz = dateutil.tz.gettz(local_timezone)
    self.tesla_api = Tesla(username, verify=True)
    # TODO(): Add some error handling here.
    # self.tesla.authorized
    self.tesla_api.fetch_token()
    self.battery = self.tesla_api.battery_list()[0]

  def setup(self):
    """ Idempotent setup function for creating the SQL tables. 

      While Tesla provides the data per 5 minute period, we average that data and
      store it per hour to simplify everything else.
    """
    cur = self.con.cursor()
    # Values stored are kwh used or created for that hour.
    cur.execute('''
        CREATE TABLE IF NOT EXISTS powerwall (dayhour INTEGER PRIMARY KEY,
                solar_power REAL,
                battery_power REAL,
                grid_power REAL);''')

  def backfill_data(self):
    """ Backfill our database with the power data since installation date. """
    self.collect_data(
      datetime.fromisoformat(
        self.battery.get_battery_data()['installation_date']))

  def collect_data(self, start_date=None):
    """ Collects data from start_date (defaults to 7 days ago.) to now.

      Data newer than a week ago will be refreshed even if we have it, data
      older than a week ago will only be retrieved from Tesla if we don't have
      the data. Otherwise this repreents a lot of Tesla API calls and might get
      you temporarily blocked from their API.
    """
    energy_fields = ['solar_power', 'battery_power', 'grid_power']

    week_ago = datetime.now(tz=self.tz) - timedelta(days=7)
    if start_date:
      current_date = start_date
    else:
      current_date = week_ago

    yesterday = datetime.now(tz=self.tz) - timedelta(days=1)

    # Used to count sql updates in a transaction as to not have crazy large sql transactions.
    n = 0
    cur = self.con.cursor()

    while current_date < yesterday:
      current_date += timedelta(days=1)
      hourly = {}
      cur.execute(
        ''' SELECT COUNT(*)
                      FROM powerwall
                      WHERE dayhour >= ? AND dayhour <= ? ''',
        (current_date.strftime("%Y%m%d00"), current_date.strftime("%Y%m%d23")))
      row_count = cur.fetchall()

      # We already have all the data we need for this older data. For data from
      # the last week we will retrieve it anyway to check it hasn't been updated.
      if (len(row_count) == 1 and row_count[0][0] == 24 and
          current_date < week_ago):
        continue

      start_of_day = datetime(current_date.year,
                              current_date.month,
                              current_date.day,
                              0,
                              0,
                              0,
                              tzinfo=self.tz)
      end_of_day = datetime(current_date.year,
                            current_date.month,
                            current_date.day,
                            23,
                            59,
                            59,
                            tzinfo=self.tz)

      battery_timeseries = self.battery.get_calendar_history_data(
        kind="power",
        period="day",
        start_date=start_of_day.isoformat(),
        end_date=end_of_day.isoformat(),
        timezone=self.tz)

      # No telsa time series data for this day.
      if 'time_series' not in battery_timeseries:
        continue

      logger.info("Storing Tesla Powerwall power data for: %s",
                  format_datetime(start_of_day))

      for timestamp in battery_timeseries['time_series']:
        time_key = parse(timestamp['timestamp']).strftime("%Y%m%d%H")
        if time_key not in hourly:
          hourly[time_key] = {}
          for kind in energy_fields:
            hourly[time_key][kind] = 0.0
        for kind in energy_fields:
          # Tesla returns in lots of 5 minutes. So sum 1/12 of each to
          # determine the mean kilowatts for the hour.
          # TODO: We really should record battery charge/discharge, grid
          # usage/export seperately since both conditions can happen in the same
          # hour, eg sporadic clouds.
          hourly[time_key][kind] += timestamp[kind] / 12.0

      for time_key in hourly:
        n = n + 1
        if n == 100:
          self.con.commit()
          cur = self.con.cursor()
          n = 0
        sql = ''' INSERT OR REPLACE INTO
                  powerwall(dayhour, solar_power, battery_power, grid_power)
                  VALUES(?,?,?,?) '''
        cur.execute(
          sql,
          (time_key, hourly[time_key]['solar_power'],
           hourly[time_key]['battery_power'], hourly[time_key]['grid_power']))
        self.con.commit()

    # Limit to 30 days of data retrieval every minute, you don't want Tesla
    # blocking you from the API (sometimes the block will be 24 hours).
    @sleep_and_retry
    @limits(calls=30, period=60)
    def get_calendar_history_data(self, **args):
      return self.battery.get_calendar_history_data(args)
