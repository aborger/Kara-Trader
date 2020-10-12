import schedule
import time
import os

def buy():
   print("Running AI...")
   os.system("sudo python3 tradeAI.py buy")
   return
   
def trail():
   print("Running AI...")
   os.system("sudo python3 tradeAI.py trail")
   return
   
def log():
   print("Running AI...")
   os.system("sudo python3 tradeAI.py log")
   return

schedule.every().day.at("13:00").do(buy)
schedule.every().day.at("07:30").do(trail)
#schedule.every().hour.do(log)

while True:
   print('Waiting...')
   schedule.run_pending()
   time.sleep(60)
