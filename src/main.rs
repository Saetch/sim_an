use rand::Rng;
use rand::seq::SliceRandom;
fn main() {

    let cities_vec = vec![(5.0, 10.0), (30.0, 60.0), (50.0, 80.0), (10.0, 73.0), (25.0, 3.0), ( 17.0 , 67.0), (98.0, 12.0), (37.0, 23.0), (46.0, 62.0), (22.4, 44.0), (41.3, 87.2)];

    let tsp = TravelingSalesmanProblem {
        cities: cities_vec,
    };
    let mut sa = SimulatedAnnealing::new(&tsp, 100.0, 0.003);
    let state = sa.optimize();
    //rotate the vector to start from the first city
    let state = state.iter().cycle().skip_while(|&&x| x != 0).take(state.len()).cloned().collect::<Vec<usize>>();

    
    println!("\nOptimal state: {:?}", state);
    println!("Optimal energy: {}\n", tsp.energy(&state));

}

// Define a trait for the optimization problem
pub trait OptimizationProblem {
    // Define the type of the state and the energy/cost function
    type State;
    fn initial_state(&self) -> Self::State;
    fn energy(&self, state: &Self::State) -> f64;
    fn neighbor(&self, state: &Self::State) -> Self::State;
}


#[derive(Debug)]
pub struct SimulatedAnnealing<'a, T: OptimizationProblem> {
    problem: &'a T,
    temperature: f64,
    cooling_rate: f64,
}

impl<'a, T: OptimizationProblem> SimulatedAnnealing<'a, T> {
    fn new(problem: &'a T, initial_temperature: f64, cooling_rate: f64) -> Self {
        SimulatedAnnealing {
            problem,
            temperature: initial_temperature,
            cooling_rate,
        }
    }


    fn optimize(&mut self) -> T::State {
        let mut current_state = self.problem.initial_state();
        let mut rng = rand::thread_rng();

        while self.temperature > 0.001 {
            for _ in 0..150 {
                let new_state = self.problem.neighbor(&current_state);
                let current_energy = self.problem.energy(&current_state);
                let new_energy = self.problem.energy(&new_state);

                if new_energy < current_energy || rng.gen::<f64>() < f64::exp((current_energy - new_energy) / self.temperature) {
                    current_state = new_state;
                }
            }

            // Condition to adjust the temperature
            if rng.gen::<f64>() < 0.5 {
                self.temperature *= 1.0 - self.cooling_rate;
            }
        }
        current_state
    }
}

// Example implementation of the OptimizationProblem trait
#[derive(Debug)]
struct ExampleProblem{
    optimal_state: f64,
}

impl OptimizationProblem for ExampleProblem {
    type State = f64;

    fn initial_state(&self) -> f64 {
        0.0
    }

    fn energy(&self, state: &f64) -> f64 {
        // Example energy function (minimize the square of the difference from 5.0)
        (state - self.optimal_state).powi(2)
    }

    fn neighbor(&self, state: &f64) -> f64 {
        // Example neighbor function (add a small random value to the state)
        state + rand::thread_rng().gen_range(-1.0..1.0)
    }
}

struct TravelingSalesmanProblem {
    cities: Vec<(f64, f64)>,
}

impl OptimizationProblem for TravelingSalesmanProblem{
    type State = Vec<usize>;

    fn initial_state(&self) -> Vec<usize> {
        let mut state: Vec<usize> = (0..self.cities.len()).collect();
        state.shuffle(&mut rand::thread_rng());
        state
    }

    fn energy(&self, state: &Vec<usize>) -> f64 {
        let mut energy = 0.0;
        for i in 0..state.len() {
            let (x1, y1) = self.cities[state[i]];
            let (x2, y2) = self.cities[state[(i + 1) % state.len()]];
            energy += ((x1 - x2).powi(2) + (y1 - y2).powi(2)).sqrt();
        }
        energy
    }

    fn neighbor(&self, state: &Vec<usize>) -> Vec<usize> {
        let mut new_state = state.clone();
        let i = rand::thread_rng().gen_range(0..new_state.len());
        let j = rand::thread_rng().gen_range(0..new_state.len());
        new_state.swap(i, j);
        new_state
    }
}