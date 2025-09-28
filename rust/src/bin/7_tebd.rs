// (Simplified) TEBD simulation of random quantum circuits

use anyhow::Result;
use ndarray::{Array, Array1, Array2, Array3, Array4, ArrayD, Axis, Ix2, array, s};
use ndarray_einsum::einsum;
use ndarray_linalg::Norm;
use num_complex::Complex64 as C64;
use rand::Rng;
use rand::distributions::{Distribution, Uniform};
use tn_basics::{EasySVD, MapStrToAnyhowErr};

fn iexp(theta: f64) -> C64 {
    C64::new(theta.cos(), theta.sin())
}

// // fn random_u(rng: &mut impl Rng) -> Array2<C64> {
// // }

// // 全状態テンソル state(2,2,...,2) に pos(0-indexed) の1量子ゲートUを適用
// fn apply_1q(state: &mut ArrayD<C64>, pos: usize, u: &Array2<C64>) {
//     let n = state.ndim();
//     let mut order: Vec<usize> = (0..n).collect();
//     order.swap(0, pos);
//     let inv = {
//         let mut inv = vec![0; n];
//         for (i, &o) in order.iter().enumerate() {
//             inv[o] = i;
//         }
//         inv
//     };
//     let st = state.view().permuted_axes(order);
//     let mut st2 = st.to_shared().into_shape((2, st.len() / 2)).unwrap();
//     st2 = u.dot(&st2);
//     let st_back = st2.into_shape(st.raw_dim()).unwrap();
//     *state = st_back.permuted_axes(inv).to_shared().into_dyn();
// }

// // state に CNOT(2,2,2,2) を pos,pos+1 に適用
// fn apply_2q(state: &mut ArrayD<C64>, pos: usize, g: &Array4) {
//     let n = state.ndim();
//     let mut order: Vec<usize> = (0..n).collect();
//     order.swap(0, pos);
//     order.swap(1, pos + 1);
//     let inv = {
//         let mut inv = vec![0; n];
//         for (i, &o) in order.iter().enumerate() {
//             inv[o] = i;
//         }
//         inv
//     };
//     let st = state.view().permuted_axes(order);
//     let mut st2 = st.to_shared().into_shape((2, 2, st.len() / 4)).unwrap();
//     let tmp: Array3<C64> = einsum("abxy,xyc->abc", &[g, &st2]).unwrap();
//     let st_back = tmp.into_shape(st.raw_dim()).unwrap();
//     *state = st_back.permuted_axes(inv).to_shared().into_dyn();
// }

// fn mps_two_site_update(
//     mps_left: &Array3<C64>,
//     mps_right: &Array3<C64>,
//     g: &Array4<C64>,
//     cutoff: f64,
//     chi_max: usize,
// ) -> (Array3<C64>, Array3<C64>) {
//     // T[χl,i1,i2,χr] = G[i1,i2,j1,j2] A[χl,j1,α] B[α,j2,χr]
//     let t: ndarray::Array4<C64> =
//         einsum("i1i2j1j2,lj1a,aj2r->li1i2r", &[g, mps_left, mps_right]).unwrap();
//     let (χl, _, _, χr) = (t.shape()[0], t.shape()[1], t.shape()[2], t.shape()[3]);
//     let tmat = t.into_shape((χl * 2, 2 * χr)).unwrap();

//     let (u, s, vt) = tmat.thin_svd().unwrap();

//     let mut r = 0usize;
//     for &sv in s.iter() {
//         if sv.re > cutoff {
//             r += 1;
//         } else {
//             break;
//         }
//         if r == chi_max {
//             break;
//         }
//     }
//     if r == 0 {
//         return (
//             Array3::<C64>::zeros((χl, 2, 1)),
//             Array3::<C64>::zeros((1, 2, χr)),
//         );
//     }
//     let s_keep = s.slice(s![0..r]).to_owned();
//     u = u.slice(s![.., 0..r]).to_owned();
//     vt = vt.slice(s![0..r, ..]).to_owned();

//     // Anew = U * sqrt(S),  Bnew = sqrt(S) * Vt
//     let mut s_sqrt = Array2::<C64>::zeros((r, r));
//     for i in 0..r {
//         s_sqrt[(i, i)] = C64::new(s_keep[i].re.sqrt(), 0.0);
//     }
//     let anew = u.dot(&s_sqrt);
//     let bnew = s_sqrt.dot(&vt);

//     let left = anew.into_shape((χl, 2, r)).unwrap();
//     let right = bnew.into_shape((r, 2, χr)).unwrap();
//     (left, right)
// }

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

            // due to ndarray_einsum limitation, we apply gate by coonverting vector into fixed shape tensor
            let state_3d = state.to_shape((1usize << pos, 2, 1usize << (n - pos - 1)))?;
            let state_3d = einsum("ij,kjm->kim", &[&u, &state_3d]).map_str_err()?;
            state = state_3d.flatten().into_owned();

            let sub0 = einsum("ij,kjm->kim", &[&u, &mps0[pos]]).map_str_err()?;
            let [s0, s1, s2] = *sub0.shape() else {
                panic!("unexpected shape")
            };
            mps0[pos] = sub0.into_shape_with_order((s0, s1, s2))?;
            let sub1 = einsum("ij,kjm->kim", &[&u, &mps1[pos]]).map_str_err()?;
            let [s0, s1, s2] = *sub1.shape() else {
                panic!("unexpected shape")
            };
            mps1[pos] = sub1.into_shape_with_order((s0, s1, s2))?;
        }

        // cnot gates
        // for i in ((k / 2)..(n - 1)).step_by(2) {
        //     // due to ndarray_einsum limitation, we apply gate by coonverting vector into fixed shape tensor
        //     apply_2q(&mut state, i, &cnot);
        //     // no trunc
        //     let (l0, r0) = mps_two_site_update(&mps0[i], &mps0[i + 1], &cnot, cutoff, usize::MAX);
        //     mps0[i] = l0;
        //     mps0[i + 1] = r0;
        //     // trunc
        //     let (l1, r1) = mps_two_site_update(&mps1[i], &mps1[i + 1], &cnot, cutoff, max_dim);
        //     mps1[i] = l1;
        //     mps1[i + 1] = r1;
        // }

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
    for i in 0..n {
        print!("{}, ", mps0[i].shape()[2]);
    }
    println!("]\n");
    print!("mps1 final virtual bond dimensions: [",);
    for i in 0..n {
        print!("{}, ", mps1[i].shape()[2]);
    }
    println!("]\n");
    Ok(())
}
