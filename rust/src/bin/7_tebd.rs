// (Simplified) TEBD simulation of random quantum circuits

use anyhow::Result;
use ndarray::{array, s, Array, Array1, Array2, Array3, ArrayD, Axis, Ix2};
use ndarray_einsum::einsum;
use ndarray_linalg::svd::SVD;
use num_complex::Complex64 as C64;
use rand::Rng;

fn cis(x: f64) -> C64 { C64::new(x.cos(), x.sin()) }

fn random_u(rng: &mut impl Rng) -> Array2<C64> {
    let alpha = 2.0 * std::f64::consts::PI * rng.gen::<f64>();
    let theta = std::f64::consts::PI * rng.gen::<f64>();
    let phi   = 2.0 * std::f64::consts::PI * rng.gen::<f64>();
    array![
        [C64::from(theta/2.0).cos(),     -cis(phi) * C64::from(theta/2.0).sin()],
        [cis(-phi) * C64::from(theta/2.0).sin(),  C64::from(theta/2.0).cos()]
    ] * cis(alpha)
}

// 全状態テンソル state(2,2,...,2) に pos(0-indexed) の1量子ゲートUを適用
fn apply_1q(state: &mut ArrayD<C64>, pos: usize, u: &Array2<C64>) {
    let n = state.ndim();
    let mut order: Vec<usize> = (0..n).collect();
    order.swap(0, pos);
    let inv = {
        let mut inv = vec![0; n];
        for (i,&o) in order.iter().enumerate(){ inv[o]=i; }
        inv
    };
    let st = state.view().permuted_axes(order);
    let mut st2 = st.to_shared().into_shape((2, st.len()/2)).unwrap();
    st2 = u.dot(&st2);
    let st_back = st2.into_shape(st.raw_dim()).unwrap();
    *state = st_back.permuted_axes(inv).to_shared().into_dyn();
}

// state に CNOT(2,2,2,2) を pos,pos+1 に適用
fn apply_2q(state: &mut ArrayD<C64>, pos: usize, g: &Array4) {
    let n = state.ndim();
    let mut order: Vec<usize> = (0..n).collect();
    order.swap(0, pos);
    order.swap(1, pos+1);
    let inv = {
        let mut inv = vec![0; n];
        for (i,&o) in order.iter().enumerate(){ inv[o]=i; }
        inv
    };
    let st = state.view().permuted_axes(order);
    let mut st2 = st.to_shared().into_shape((2,2, st.len()/4)).unwrap();
    let tmp: Array3<C64> = einsum("abxy,xyc->abc", &[g, &st2]).unwrap();
    let st_back = tmp.into_shape(st.raw_dim()).unwrap();
    *state = st_back.permuted_axes(inv).to_shared().into_dyn();
}

type Array4 = ndarray::Array4<C64>;

fn mps_two_site_update(
    mps_left: &Array3<C64>,
    mps_right: &Array3<C64>,
    g: &Array4,
    cutoff: f64,
    chi_max: usize,
) -> (Array3<C64>, Array3<C64>) {
    // T[χl,i1,i2,χr] = G[i1,i2,j1,j2] A[χl,j1,α] B[α,j2,χr]
    let t: ndarray::Array4<C64> = einsum("i1i2j1j2,lj1a,aj2r->li1i2r", &[g, mps_left, mps_right]).unwrap();
    let (χl, _, _, χr) = (t.shape()[0], t.shape()[1], t.shape()[2], t.shape()[3]);
    let tmat = t.into_shape((χl*2, 2*χr)).unwrap();

    let (u_opt, s, vt_opt) = tmat.svd(true, true).unwrap();
    let (mut u, mut vt) = (u_opt.unwrap(), vt_opt.unwrap());

    let mut r = 0usize;
    for &sv in s.iter() {
        if sv.re > cutoff { r += 1; } else { break; }
        if r == chi_max { break; }
    }
    if r == 0 {
        return (
            Array3::<C64>::zeros((χl, 2, 1)),
            Array3::<C64>::zeros((1,  2, χr))
        );
    }
    let s_keep = s.slice(s![0..r]).to_owned();
    u = u.slice(s![.., 0..r]).to_owned();
    vt = vt.slice(s![0..r, ..]).to_owned();

    // Anew = U * sqrt(S),  Bnew = sqrt(S) * Vt
    let mut s_sqrt = Array2::<C64>::zeros((r, r));
    for i in 0..r { s_sqrt[(i,i)] = C64::new(s_keep[i].re.sqrt(), 0.0); }
    let anew = u.dot(&s_sqrt);
    let bnew = s_sqrt.dot(&vt);

    let left = anew.into_shape((χl, 2, r)).unwrap();
    let right = bnew.into_shape((r, 2, χr)).unwrap();
    (left, right)
}

fn main() -> Result<()> {
    let n: usize = 16;
    let depth: usize = 16;
    let max_dim: usize = 4;
    let cutoff: f64 = 1e-10;

    // CNOT
    let mut cnot = Array::from_elem((2,2,2,2), C64::new(0.0,0.0));
    cnot[[0,0,0,0]] = 1.0.into();
    cnot[[0,1,0,1]] = 1.0.into();
    cnot[[1,0,1,1]] = 1.0.into();
    cnot[[1,1,1,0]] = 1.0.into();

    // |0...0>
    let mut state = ArrayD::<C64>::zeros(vec![2; n]);
    state[[vec![0; n].as_slice()].concat()] = 1.0.into();

    // MPS init
    let init = Array3::from_shape_vec((1,2,1), vec![1.0.into(), 0.0.into()])?;
    let mut mps0: Vec<Array3<C64>> = (0..n).map(|_| init.clone()).collect();
    let mut mps1: Vec<Array3<C64>> = (0..n).map(|_| init.clone()).collect();
    println!("mps0/1 initial virtual bond dimensions: {:?}", (0..n-1).map(|i| mps0[i].dim().2).collect::<Vec<_>>());

    let mut rng = rand::thread_rng();

    for k in 0..depth {
        // 1-qubit random rotations
        for pos in 0..n {
            let u = random_u(&mut rng);
            apply_1q(&mut state, pos, &u);
            // mps update: "ij,kjm->kim"
            let left0: Array3<C64> = einsum("ij,kjm->kim", &[&u, &mps0[pos]]).unwrap();
            let left1: Array3<C64> = einsum("ij,kjm->kim", &[&u, &mps1[pos]]).unwrap();
            mps0[pos] = left0;
            mps1[pos] = left1;
        }

        // CNOT layer
        for i in ((k/2)..(n-1)).step_by(2) {
            apply_2q(&mut state, i, &cnot);
            // no trunc
            let (l0, r0) = mps_two_site_update(&mps0[i], &mps0[i+1], &cnot, cutoff, usize::MAX);
            mps0[i] = l0; mps0[i+1] = r0;
            // trunc
            let (l1, r1) = mps_two_site_update(&mps1[i], &mps1[i+1], &cnot, cutoff, max_dim);
            mps1[i] = l1; mps1[i+1] = r1;
        }

        // reconstruct from MPS (mps0)
        let mut ψ0 = mps0[0].clone().into_shape((2, mps0[0].dim().2))?.to_owned();
        for i in 1..n {
            let tmp: Array3<C64> = einsum("ij,jkl->ikl", &[&ψ0, &mps0[i]]).unwrap();
            let (a,b,c) = tmp.dim();
            ψ0 = tmp.into_shape((a*b, c))?;
        }
        let ψ0_vec: Vec<C64> = ψ0.into_raw_vec();

        // reconstruct from MPS (mps1)
        let mut ψ1 = mps1[0].clone().into_shape((2, mps1[0].dim().2))?.to_owned();
        for i in 1..n {
            let tmp: Array3<C64> = einsum("ij,jkl->ikl", &[&ψ1, &mps1[i]]).unwrap();
            let (a,b,c) = tmp.dim();
            ψ1 = tmp.into_shape((a*b, c))?;
        }
        let ψ1_vec: Vec<C64> = ψ1.into_raw_vec();

        // flatten full state (row-major)
        let st_mat = state.clone().into_shape((2, state.len()/2))?;
        let st_vec: Vec<C64> = st_mat.into_raw_vec();

        // fidelity
        let dot0: C64 = st_vec.iter().zip(ψ0_vec.iter()).map(|(a,b)| a.conj()*b).sum();
        let dot1: C64 = st_vec.iter().zip(ψ1_vec.iter()).map(|(a,b)| a.conj()*b).sum();
        let fid0 = (dot0.norm_sqr()) as f64;
        let fid1 = (dot1.norm_sqr()) as f64;
        println!("step: {} fidelity: {}, {}", k, fid0, fid1);
    }

    println!("mps0 final virtual bond dimensions: {:?}", (0..n-1).map(|i| mps0[i].dim().2).collect::<Vec<_>>());
    println!("mps1 final virtual bond dimensions: {:?}", (0..n-1).map(|i| mps1[i].dim().2).collect::<Vec<_>>());
    Ok(())
}
