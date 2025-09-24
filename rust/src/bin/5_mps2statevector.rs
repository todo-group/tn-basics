use ndarray::{array, Array2, Array3, Axis};
use ndarray_einsum_beta::einsum;

fn main() {
    // Bell state
    println!("Bell state:");
    let s = 2f64.powf(0.25);
    let tl: Array2<f64> = array![[1.0, 0.0], [0.0, 1.0]] / s;
    let tr: Array2<f64> = array![[1.0, 0.0], [0.0, 1.0]] / s;
    println!("left tensor: \n{:?}", tl);
    println!("right tensor: \n{:?}\n", tr);

    let bell: Array2<f64> = einsum("ij,jk->ik", &[&tl, &tr]).unwrap();
    // Python の reshape(-1) と同順（row-major）で 1D 化
    let state_bell = bell.into_raw_vec();
    println!("statevector: {:?}\n", state_bell);

    // GHZ state
    let n: usize = 6;
    println!("n={} GHZ state:", n);
    let w = 2f64.powf(1.0 / (2.0 * n as f64));
    let tl: Array2<f64> = array![[1.0, 0.0], [0.0, 1.0]] / w;
    let tr: Array2<f64> = array![[1.0, 0.0], [0.0, 1.0]] / w;

    // t(j,k,l)
    let mut t: Array3<f64> = Array3::zeros((2, 2, 2));
    t[[0, 0, 0]] = 1.0 / w;
    t[[1, 1, 1]] = 1.0 / w;

    println!("left tensor: \n{:?}", tl);
    println!("right tensor: \n{:?}", tr);
    println!("middle tensors: \n{:?}\n", t);

    // ghz: まず (i,j) = tl
    let mut ghz: Array2<f64> = tl.clone();

    // Python: for k in range(1, n-1): ghz(i,j) × t(j,k,l) -> (i,k,l) -> reshape (i*k, l)
    for _ in 1..(n - 1) {
        let tmp: Array3<f64> = einsum("ij,jkl->ikl", &[&ghz, &t]).unwrap();
        let (i, k, l) = (tmp.shape()[0], tmp.shape()[1], tmp.shape()[2]);
        ghz = tmp.into_shape((i * k, l)).unwrap();
    }

    // 最後に右端と縮約: ghz(i,j) × tr(j,k) -> (i,k) → 1D
    let ghz2: Array2<f64> = einsum("ij,jk->ik", &[&ghz, &tr]).unwrap();
    let state_ghz = ghz2.into_raw_vec();
    println!("statevector: {:?}\n", state_ghz);
}
