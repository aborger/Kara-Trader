import schedule
import time
import os

def buy():
   print("Buying...")
   os.system("sudo python3 tradeAI.py buy")
   return
   
def trail():
   print("Trailing...")
   os.system("sudo python3 tradeAI.py trail")
   return
   
def charge():
	print("Charging...")
	os.system("sudo python3 tradeAI.py charge")
	return
   
def log():
   print("Running AI...")
   os.system("sudo python3 tradeAI.py log")
   return

schedule.every().day.at("13:00").do(buy)
schedule.every().day.at("14:00").do(
schedule.every().day.at("14:30").do(trail)
#schedule.every().hour.do(log)

while True:
   print('Waiting...')
   schedule.run_pending()
   time.sleep(60)
