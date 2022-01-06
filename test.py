
import datetime

from powerwallrl.powerplans import Powerplan
import powerwallrl.powerplans.powerplan
from powerwallrl.powerplans.australia.wa.synergy import SmartHome_Rebs


plan = SmartHome_Rebs()
print(plan.usage(datetime.datetime.now()))

print(powerwallrl.powerplans.registry)

