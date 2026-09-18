(define (problem DLOG-2-2-2)
	(:domain driverlog)
	(:objects
	driver1 - driver
	driver2 - driver
	truck1 - truck
	truck2 - truck
	package1 - obj
	package2 - obj
	s0 - location
	s1 - location
	s2 - location
	p1-0 - location
	p1-2 - location
	)
	(:init
	(loc_driver driver1 s2)
	(loc_driver driver2 s2)
	(loc_truck truck1 s0)
	(empty truck1)
	(loc_truck truck2 s0)
	(empty truck2)
	(loc_package package1 s0)
	(loc_package package2 s0)
	(path s1 p1-0)
	(path p1-0 s1)
	(path s0 p1-0)
	(path p1-0 s0)
	(path s1 p1-2)
	(path p1-2 s1)
	(path s2 p1-2)
	(path p1-2 s2)
	(link s0 s1)
	(link s1 s0)
	(link s0 s2)
	(link s2 s0)
	(link s2 s1)
	(link s1 s2)
)
	(:goal (and
	(loc_driver driver1 s1)
	(loc_truck truck1 s1)
	(loc_package package1 s0)
	(loc_package package2 s0)
	))

(:metric minimize (total-time))

)
