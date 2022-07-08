""" This module provides a class for collecting weather data for a given
lat/long and storing it in in 3 tables. One for the most up to date data, one
with the data 24 hours before and one with the first weather forcase we
collected. 

We do this because it may be helpful for varying the training model inputs based
on real world weather accuracy rather than last minute weather accuracy.
"""

import dateutil.tz
import logging
import json
import os
import sqlite3
import requests
import locale

from datetime import datetime
from babel.dates import format_datetime

logger = logging.getLogger(__name__)


class WeatherData(object):

  def __init__(self,
               api_key,
               latitude,
               longitude,
               database,
               local_timezone='Etc/UTC'):
    self.con = database
    self.api_key = api_key
    self.tz = dateutil.tz.gettz(local_timezone)
    self.latitude = latitude
    self.longitude = longitude

  def setup(self):
    """Idempotent setup function for creating the SQL tables we need.

      We keep three versions of the weather forecast incase thats useful for understanding 

    """
    cur = self.con.cursor()

    # This is roughly what our usual decision making data for given dayhour.
    cur.execute('''
      CREATE TABLE IF NOT EXISTS weather_24 (dayhour INTEGER PRIMARY KEY,
                                             temp REAL,
                                             uvi REAL,
                                             clouds INTEGER,
                                             humidity INTEGER);''')

    # The worst case / earliest forecast we recieved for a given dayhour.
    cur.execute('''
      CREATE TABLE IF NOT EXISTS weather_first (dayhour INTEGER PRIMARY KEY,
                                                temp REAL,
                                                uvi REAL,
                                                clouds INTEGER,
                                                humidity INTEGER);''')

    # This was the most update to date forecast for the hour.
    cur.execute('''
      CREATE TABLE IF NOT EXISTS weather_last (dayhour INTEGER PRIMARY KEY,
                                               temp REAL,
                                               uvi REAL,
                                               clouds INTEGER,
                                               humidity INTEGER);''')

  def weather(self):
    url = ("https://api.openweathermap.org/data/2.5/onecall?lat=%s&lon=%s"
           "&exclude=current,minutely,alerts,daily&appid=%s&units=metric") % (
             self.latitude, self.longitude, self.api_key)

    response = requests.get(url)
    weather_data = json.loads(response.text)
    logging.debug("Weather data: %s", weather_data)
    return weather_data

  def save_weather_data(self, weather_data=None, collection_time=None):
    cur = self.con.cursor()
    if not weather_data:
      weather_data = self.weather()
    if not collection_time:
      collection_time = datetime.now(tz=self.tz)

    logger.info("Collecting weather data as of: %s",
                format_datetime(collection_time))

    for hour_dict in weather_data['hourly']:
      h = datetime.fromtimestamp(hour_dict['dt'], self.tz).strftime('%Y%m%d%H')
      sql = ''' INSERT OR REPLACE INTO weather_last(dayhour,temp,uvi,clouds,humidity)
              VALUES(?,?,?,?,?) '''
      cur.execute(sql, (h, hour_dict['temp'], hour_dict['uvi'],
                        int(hour_dict['clouds']), int(hour_dict['humidity'])))
      # Stop inserting fresh data into weather_24 once we are recieving forecasts inside of 24 hours.
      if int(collection_time.strftime('%Y%m%d%H')) < (int(h) - 23):
        sql = ''' INSERT OR REPLACE INTO weather_24(dayhour,temp,uvi,clouds,humidity)
                VALUES(?,?,?,?,?) '''
        cur.execute(sql, (h, hour_dict['temp'], hour_dict['uvi'],
                          int(hour_dict['clouds']), int(hour_dict['humidity'])))
      sql = ''' INSERT OR IGNORE INTO weather_first(dayhour,temp,uvi,clouds,humidity)
              VALUES(?,?,?,?,?) '''
      cur.execute(sql, (h, hour_dict['temp'], hour_dict['uvi'],
                        int(hour_dict['clouds']), int(hour_dict['humidity'])))
    self.con.commit()
    logger.info("Weather data commited to database.")
