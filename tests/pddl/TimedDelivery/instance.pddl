(define (problem timed-delivery-1)
  (:domain timed-delivery)
  (:objects
    p1 p2 - package
    depot shop market - location)
  (:init
    (at p1 depot)
    (at p2 depot)
    (road depot shop)
    (road depot market)
    (free)
    (open depot)
    (at 4 (open shop))
    (at 9 (not (open shop)))
    (at 2.5 (open market))
    (at 12 (not (open market))))
  (:goal (and
    (at p1 shop)
    (at p2 market))))
