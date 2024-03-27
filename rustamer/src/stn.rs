use std::collections::HashMap;
use std::collections::VecDeque;
use std::fs::read_to_string;
use std::sync::Arc;

use regex::Regex;


#[derive(Debug)]
struct DeltaNeighbors<Q> {
    dst: i32,
    bound: Q,
    next: Option<Arc<DeltaNeighbors<Q>>>,
}

impl<Q> DeltaNeighbors<Q>
where Q: Clone {
    fn mk_empty() -> Option<Arc<Self>> {
        None
    }

    fn add(dst: &i32,
        bound: &Q,
        next: &Option<Arc<Self>>) -> Option<Arc<Self>> {
        Some(Arc::new(DeltaNeighbors { dst:*dst, bound:bound.clone(), next:next.clone() }))
    }
}

#[derive(Debug)]
pub struct DeltaSTN<Q> {
    constraints: HashMap<i32, Option<Arc<DeltaNeighbors<Q>>>>,
    pub distances: HashMap<i32, Q>,
    is_sat: bool,
    pub tolerance: Q,
}

impl<Q> Clone for DeltaSTN<Q> where Q: Clone {
    fn clone(&self) -> Self {
        DeltaSTN {
            constraints: self.constraints.clone(),
            distances: self.distances.clone(),
            is_sat: self.is_sat,
            tolerance: self.tolerance.clone(),
        }
    }
}

impl<Q> DeltaSTN<Q>
where
Q: num_traits::Num + std::ops::Neg<Output=Q> + PartialOrd + Clone {
    pub fn new(tolerance: Q) -> Self {
        DeltaSTN {
            constraints: HashMap::new(),
            distances: HashMap::new(),
            is_sat: true,
            tolerance: tolerance,
        }
    }

    pub fn add(& mut self, x:&i32, y:&i32, b:&Q) {
        if self.is_sat {
            if !self.distances.contains_key(x) {
                self.distances.insert(*x, Q::zero());
                self.constraints.insert(*x, DeltaNeighbors::mk_empty());
            }
            if !self.distances.contains_key(&y) {
                self.distances.insert(*y, Q::zero());
                self.constraints.insert(*y, DeltaNeighbors::mk_empty());
            }
            if !self.is_subsumed(&x, &y, &b) {
                let old_x= self.constraints.get(&x).unwrap();
                self.constraints.insert(x.clone(), DeltaNeighbors::add(&y, &b, old_x));
            }
            self.is_sat = self.inc_check(&x, &y, &b);
        }
    }

    pub fn check(&self) -> bool {
        self.is_sat
    }

    pub fn get_model_value(&self, x:&i32) -> Option<Q> {
        self.distances.get(x).map(|v| v.clone() * (- Q::one()))
    }

    fn is_subsumed(&self, x:&i32, y:&i32, b:&Q) -> bool {
        let mut neighbors: &Option<Arc<DeltaNeighbors<Q>>> = self.constraints.get(x).unwrap();
        while neighbors.is_some() {
            let n: &Arc<DeltaNeighbors<Q>> = neighbors.as_ref().unwrap();
            if n.dst == *y {
                return n.bound <= b.clone() + self.tolerance.clone()
            }
            neighbors = &n.next
        }
        false
    }

    pub fn equals_with_tolerance(&self, b1: &Q, b2: &Q) -> bool {
        if b1.clone() - b2.clone() <= self.tolerance && b1.clone() - b2.clone() >= -self.tolerance.clone() {
            true
        } else {
            false
        }
    }

    fn inc_check(&mut self, x:&i32, y:&i32, b:&Q) -> bool {
        if self.distances[x].clone() + b.clone() < self.distances[y].clone() - self.tolerance.clone() {
            self.distances.insert(*y, self.distances[x].clone() + b.clone());
        }
        else {
            return true;
        }

        let mut q: VecDeque<&i32> = VecDeque::from([y]);
        while ! q.is_empty() {
            let c: &i32 = q.pop_front().unwrap();
            let mut neighbors: &Option<Arc<DeltaNeighbors<Q>>> = self.constraints.get(c).unwrap();
            while neighbors.is_some() {
                let n: &Arc<DeltaNeighbors<Q>> = neighbors.as_ref().unwrap();
                let val = self.distances[c].clone() + n.bound.clone();
                if val < self.distances[&n.dst].clone() - self.tolerance.clone() {
                    if n.dst == *y && self.equals_with_tolerance(&n.bound, b) {
                        return false; // Cycle detected
                    }
                    else {
                        self.distances.insert(n.dst, val);
                        q.push_back(&n.dst);
                    }
                }
                neighbors = &n.next
            }
        }
        true
    }

}


pub fn _tnsolve(fname: String) -> () {
    let re_new_tn = Regex::new(r#"^NewTN\("([a-z0-9]+)"\);$"#).unwrap();
    let re_check = Regex::new(r#"^Check\("([a-z0-9]+)"\);$"#).unwrap();
    let re_destroy_tn = Regex::new(r#"^DestroyTN\("([a-z0-9]+)"\);$"#).unwrap();
    let re_copy_tn = Regex::new(r#"^CopyTN\("([a-z0-9]+)",\s*"([a-z0-9]+)"\);$"#).unwrap();
    let re_add = Regex::new(r#"^Add\("([a-z0-9]+)",\s*([0-9]+),\s*([0-9]+),\s*((-?)(0|([1-9][0-9]*))(\.[0-9]+)?)\);$"#).unwrap();

    let mut tn_map = HashMap::<String, DeltaSTN<f64>>::new();

    for line in read_to_string(fname).unwrap().lines() {
        if let Some(new_tn) = re_new_tn.captures(line) {
            tn_map.insert(new_tn[1].to_owned(), DeltaSTN::new(0.00000001));
            continue;
        }

        if let Some(check) = re_check.captures(line) {
            print!("{} ", if tn_map[&check[1].to_owned()].check() {"1"} else {"0"});
            continue;
        }

        if let Some(destroy_tn) = re_destroy_tn.captures(line) {
            tn_map.remove(&destroy_tn[1].to_owned());
            continue;
        }

        if let Some(copy_tn) = re_copy_tn.captures(line)
        {
            let map = &mut tn_map;
            let new = map[&copy_tn[1].to_owned()].clone();
            map.insert(copy_tn[2].to_owned(), new);
            continue;
        }

        if let Some(add) = re_add.captures(line)
        {
            let x = add[2].parse::<i32>().unwrap();
            let y = add[3].parse::<i32>().unwrap();
            let b = add[4].parse::<f64>().unwrap();
            tn_map.get_mut(&add[1].to_owned()).unwrap().add(&x, &y, &b);
            continue;
        }

        println!("Unmatched line: {}", line)
    }
    println!("")
}
