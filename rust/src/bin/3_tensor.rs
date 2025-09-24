use ndarray::{array, Array1, Array2, Array3, s};
use rand::random;

fn main() {
    println!("vector (1-leg tensor)");
    let t1: Array1<i32> = array![1, 2, 3];
    println!("t1 = {:?}", t1);
    println!("t1.shape = {:?}", t1.shape());
    println!("t1[0] = {}", t1[0]);
    println!("t1[1] = {}\n", t1[1]);

    println!("(random) matrix (2-leg tensor)");
    let t2: Array2<f64> = Array2::from_shape_fn((2, 3), |_| random::<f64>());
    println!("t2 =\n{:?}", t2);
    println!("t2.shape = {:?}", t2.shape());
    println!("t2[0,0] = {}", t2[(0, 0)]);
    println!("t2[1,2] = {}\n", t2[(1, 2)]);

    println!("3D array (3-leg tensor)");
    let a: Array3<i32> = array![
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ];
    println!("A =\n{:?}\n", a);
    println!("A.shape = {:?}", a.shape());
    println!("A[0,0,0] = {}", a[(0,0,0)]);
    println!("A[1,1,1] = {}", a[(1,1,1)]);
    println!("A[:,0,1] = {:?}", a.slice(s![.., 0, 1]));
}
