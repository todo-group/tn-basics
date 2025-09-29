use ndarray::{Array2, Array3, ArrayD, array};
use ndarray_einsum::einsum;
use tn_basics::MapStrToAnyhowErr;

// Generate statevector from MPS

#[expect(clippy::cast_precision_loss)]
fn main() -> anyhow::Result<()> {
    // Bell state
    println!("Bell state:");
    let tl: Array2<f64> = array![[1.0, 0.0], [0.0, 1.0]] / 2f64.powf(0.25);
    let tr: Array2<f64> = array![[1.0, 0.0], [0.0, 1.0]] / 2f64.powf(0.25);
    println!("left tensor:\n{tl}");
    println!("right tensor:\n{tr}\n");

    let bell = einsum("ij,jk->ik", &[&tl, &tr]).map_str_err()?;
    let state_bell = bell.flatten();
    println!("statevector:\n{state_bell}\n");

    // GHZ state
    let n: usize = 6;
    println!("n={n} GHZ state:");
    let w = 2f64.powf(1.0 / (2.0 * n as f64));
    let tl: Array2<f64> = array![[1.0, 0.0], [0.0, 1.0]] / w;
    let tr: Array2<f64> = array![[1.0, 0.0], [0.0, 1.0]] / w;
    let mut t = Array3::zeros((2, 2, 2));
    t[[0, 0, 0]] = 1.0 / w;
    t[[1, 1, 1]] = 1.0 / w;
    println!("left tensor:\n{tl}");
    println!("right tensor:\n{tr}");
    println!("middle tensors:\n{t}\n");

    let mut ghz: Array2<f64> = tl;
    for _ in 1..(n - 1) {
        let tmp: ArrayD<f64> = einsum("ij,jkl->ikl", &[&ghz, &t]).map_str_err()?;
        ghz = tmp
            .to_shape((tmp.shape()[0] * tmp.shape()[1], tmp.shape()[2]))?
            .into_owned();
    }
    let ghz = einsum("ij,jk->ik", &[&ghz, &tr]).map_str_err()?;
    let state_ghz = ghz.flatten();
    println!("statevector:\n{state_ghz}\n");

    Ok(())
}
