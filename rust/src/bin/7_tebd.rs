use anyhow::Result;
use ndarray::{Array1, Array2, Array3, Array4, array, s};
use ndarray_einsum::einsum;
use num_complex::Complex64 as C64;
use rand::distributions::{Distribution, Uniform};
use tn_basics::{EasySVD, MapStrToAnyhowErr};

fn iexp(theta: f64) -> C64 {
    C64::new(theta.cos(), theta.sin())
}

// (Simplified) TEBD simulation of random quantum circuits

fn main() -> Result<()> {
    let n: usize = 16;
    let depth: usize = 16;
    let max_dim: usize = 4;
    let cutoff: f64 = 1e-10;

    // two-qubit gate: CNOT
    let cnot: Array4<C64> = array![
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ]
    .mapv(|x: f64| C64::from(x))
    .into_shape_with_order((2, 2, 2, 2))?;

    // state vector |0...0>
    let mut state = Array1::<C64>::zeros([1usize << n]);
    state[0] = 1.0.into();
    //let state = state.into_shape_with_order(vec![2; n])?;

    // MPS |0...0> without truncation
    let mut mps0: Vec<Array3<C64>> = (0..n)
        .map(|_| array![[[1.0.into()], [0.0.into()]]])
        .collect();
    print!("mps0/1 initial virtual bond dimensions: [",);
    for i in 0..n {
        print!("{}, ", mps0[i].shape()[2]);
    }
    println!("]\n");

    // MPS |0...0> with truncation
    let mut mps1 = mps0.clone();

    let mut rng = rand::thread_rng();
    let uniform_2pi = Uniform::new(0.0, 2.0 * std::f64::consts::PI);
    let uniform_pi = Uniform::new(0.0, std::f64::consts::PI);

    for k in 0..depth {
        // random single-qubit rotations
        for pos in 0..n {
            let alpha = uniform_2pi.sample(&mut rng);
            let theta = uniform_pi.sample(&mut rng);
            let phi = uniform_2pi.sample(&mut rng);

            let u = iexp(alpha)
                * array![
                    [
                        C64::from((theta / 2.0).cos()),
                        -iexp(phi) * C64::from((theta / 2.0).sin())
                    ],
                    [
                        iexp(-phi) * C64::from((theta / 2.0).sin()),
                        C64::from((theta / 2.0).cos())
                    ]
                ];

            {
                // due to ndarray_einsum limitation, we apply gate by coonverting vector into fixed shape tensor
                let state_3d = state.to_shape((1usize << pos, 2, 1usize << (n - pos - 1)))?;
                let state_3d = einsum("ij,kjm->kim", &[&u, &state_3d]).map_str_err()?;
                state = state_3d.flatten().into_owned();
            }
            {
                let sub0 = einsum("ij,kjm->kim", &[&u, &mps0[pos]]).map_str_err()?;
                let [s0, s1, s2] = *sub0.shape() else {
                    panic!("unexpected shape")
                };
                mps0[pos] = sub0.into_shape_clone((s0, s1, s2))?;
            }
            {
                let sub1 = einsum("ij,kjm->kim", &[&u, &mps1[pos]]).map_str_err()?;
                let [s0, s1, s2] = *sub1.shape() else {
                    panic!("unexpected shape")
                };
                mps1[pos] = sub1.into_shape_clone((s0, s1, s2))?;
            }
        }

        // cnot gates
        for i in ((k / 2)..(n - 1)).step_by(2) {
            {
                // due to ndarray_einsum limitation, we apply gate by coonverting vector into fixed shape tensor
                let state_4d = state.to_shape((1usize << i, 2, 2, 1usize << (n - i - 2)))?;
                let state_4d = einsum("ijkl,mkln->mijn", &[&cnot, &state_4d]).map_str_err()?;
                state = state_4d.flatten().into_owned();
            }
            {
                let sub0 = einsum("ijkl,mkn->mijln", &[&cnot, &mps0[i]]).map_str_err()?;
                let sub0 = einsum("ijklm,mln->ijkn", &[&sub0, &mps0[i + 1]]).map_str_err()?;
                let [s0, s1, s2, s3] = *sub0.shape() else {
                    panic!("unexpected shape")
                };
                let tmat = sub0.to_shape((s0 * s1, s2 * s3))?;
                let (u, s, vt) = tmat.thin_svd()?;
                let rank_new = s
                    .iter()
                    .position(|&x| x <= cutoff * s[0])
                    .unwrap_or(s.len());
                let s = s.slice(s![0..rank_new]);
                let smat = Array2::from_diag(&s.mapv(|x| C64::from(x.sqrt())));
                mps0[i] = (u.slice_move(s![.., 0..rank_new]))
                    .dot(&smat)
                    .into_shape_clone((s0, s1, rank_new))?;
                mps0[i + 1] = smat
                    .dot(&(vt.slice_move(s![0..rank_new, ..])))
                    .into_shape_clone((rank_new, s2, s3))?;
            }
            {
                let sub1 = einsum("ijkl,mkn->mijln", &[&cnot, &mps1[i]]).map_str_err()?;
                let sub1 = einsum("ijklm,mln->ijkn", &[&sub1, &mps1[i + 1]]).map_str_err()?;
                let [s0, s1, s2, s3] = *sub1.shape() else {
                    panic!("unexpected shape")
                };
                let tmat = sub1.to_shape((s0 * s1, s2 * s3))?;
                let (u, s, vt) = tmat.thin_svd()?;
                let rank_new = s
                    .iter()
                    .position(|&x| x <= cutoff * s[0])
                    .unwrap_or(s.len())
                    .min(max_dim);
                let s = s.slice(s![0..rank_new]);
                let smat = Array2::from_diag(&s.mapv(|x| C64::from(x.sqrt())));
                mps1[i] = (u.slice_move(s![.., 0..rank_new]))
                    .dot(&smat)
                    .into_shape_clone((s0, s1, rank_new))?;
                mps1[i + 1] = smat
                    .dot(&(vt.slice_move(s![0..rank_new, ..])))
                    .into_shape_clone((rank_new, s2, s3))?;
            }
        }

        let (s0, s1, s2) = mps0[0].dim();
        let mut state0 = mps0[0]
            .clone()
            .into_shape_with_order((s0 * s1, s2))?
            .to_owned();
        for i in 1..n {
            let state = einsum("ij,jkl->ikl", &[&state0, &mps0[i]]).unwrap();
            let shape = (state.shape()[0] * state.shape()[1], state.shape()[2]);
            state0 = state.into_shape_clone(shape)?;
        }
        let state0 = state0.flatten();

        let (s0, s1, s2) = mps1[0].dim();
        let mut state1 = mps1[0]
            .clone()
            .into_shape_with_order((s0 * s1, s2))?
            .to_owned();
        for i in 1..n {
            let state = einsum("ij,jkl->ikl", &[&state1, &mps1[i]]).unwrap();
            let shape = (state.shape()[0] * state.shape()[1], state.shape()[2]);
            state1 = state.into_shape_clone(shape)?;
        }
        let state1 = state1.flatten();

        let state_conj = state0.mapv(|x| x.conj());

        println!(
            "step: {} fidelity: {}, {}",
            k,
            state_conj.dot(&state0).norm_sqr(),
            state_conj.dot(&state1).norm_sqr(),
        );
    }

    print!("mps0 final virtual bond dimensions: [",);
    for i in 0..(n - 1) {
        print!("{}, ", mps0[i].shape()[2]);
    }
    println!("]\n");
    print!("mps1 final virtual bond dimensions: [",);
    for i in 0..(n - 1) {
        print!("{}, ", mps1[i].shape()[2]);
    }
    println!("]\n");
    Ok(())
}
