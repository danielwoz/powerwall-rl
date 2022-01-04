"""  This script will create the necessary databases for data collection and
trigger an initial backfill of the database given the available data.

Unfortunately I don't know of any free/good historical API for weather, so you
can only really wait for enough weather data to train.
"""

# Author: Daniel Williams

__version__ = '0.0.1'

import sqlite3
import logging
import sys
import teslapy
import time

from powerwallrl.data.weather import WeatherData
from powerwallrl.data.tesla import TeslaPowerwallData
from powerwallrl.settings import PowerwallRLConfig


def main():
  # Log INFO level message to stdout for the user to see progress.
  root = logging.getLogger()
  root.setLevel(logging.INFO)
  handler = logging.StreamHandler(sys.stdout)
  handler.setLevel(logging.INFO)
  formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  handler.setFormatter(formatter)
  root.addHandler(handler)

  config = PowerwallRLConfig()

  # Weather Data.
  db = sqlite3.connect(config.database_location)
  weather = WeatherData(config.openweathermap_api_key, config.latitude,
                        config.longitude, db, config.local_timezone)
  root.info("Collecting weather data.")
  weather.save_weather_data()

  # Power Data.
  powerwall = TeslaPowerwallData(config.tesla_username, db,
                                 config.local_timezone)
  root.info("Collecting powerwall data.")
  powerwall.collect_data()

  root.debug("All done.")


if __name__ == "__main__":
  main()
