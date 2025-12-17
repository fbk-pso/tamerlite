import unified_planning
from unified_planning.shortcuts import *


print("tamelite is present", "tamerlite" in get_environment().factory._engines)
print(get_environment().factory._engines)
print(get_environment().factory.engine("tamerlite"))
