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

import webview

def custom_auth(url):
    result = ['']
    window = webview.create_window('Login', url)
    def on_loaded():
        result[0] = window.get_current_url()
        if 'void/callback' in result[0].split('?')[0]:
            window.destroy()
    window.loaded += on_loaded
    webview.start()
    return result[0]

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
  root.info("Setting up and backfilling data to database: %s",
            config.database_location)

  # Weather Data.
  db = sqlite3.connect(config.database_location)
  weather = WeatherData(config.openweathermap_api_key, config.latitude,
                        config.longitude, db, config.local_timezone)
  root.info("Setting up weather data database tables.")
  weather.setup()
  root.info("Collecting weather data.")
  weather.save_weather_data()

  tesla = teslapy.Tesla(email=config.tesla_username, cache_file=config.tesla_cache_file)
  if not tesla.authorized:
    # Setup Tesla API authentication
    with teslapy.Tesla(email=config.tesla_username, cache_file=config.tesla_cache_file) as tesla:
      print('Use browser to login. Page Not Found will be shown at success.')
      print('Open this URL: ' + tesla.authorization_url())
      tesla.fetch_token(authorization_response=input(
          'Enter URL after authentication to cache authorization token '
          'locally: '))

  # Power Data.
  powerwall = TeslaPowerwallData(config.tesla_username, db,
                                 config.local_timezone,
                                 config.tesla_cache_file)
  root.info("Setting up powerwall data database tables.")
  powerwall.setup()
  root.info("Backfilling powerwall data. (Depending on how long installation "
            "date was ago, this might take quite sometime.)")
  powerwall.backfill_data()

  root.info("All setup! Now setup regular data collection.")

if __name__ == "__main__":
  main()
