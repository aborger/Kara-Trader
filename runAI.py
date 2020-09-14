import schedule
import time
import os

def run():
   print("Running AI...")
   os.system("sudo python3 tradeAI.py trade --time=1D")
   return

schedule.every().day.at("07:30").do(run)

while True:
   print('Waiting for open...')
   schedule.run_pending()
   time.sleep(60)
