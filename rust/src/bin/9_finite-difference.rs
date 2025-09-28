use anyhow::Result;
use ndarray::{Array1, Array2, Array3, Array4, ArrayD, s};
use ndarray_einsum::einsum;
use ndarray_linalg::Norm;
use plotters::prelude::{BitMapBackend, IntoDrawingArea};
use tn_basics::{
    EasySVD, MapStrToAnyhowErr,
    plot::{plot_error, plot_target_vs_qtt},
};

fn main() -> Result<()> {
    let depth: usize = 4;
    let npoints: usize = 1 << depth;
    let cutoff: f64 = 1e-10;
    let _max_rank: usize = 4;

    // target function and its derivative
    let x = Array1::<f64>::linspace(0.0, 1.0, npoints);
    let y = x.mapv(f64::exp);
    let dy = x.mapv(f64::exp);
    // let y = x.mapv(f64::cos);
    // let dy = x.mapv(|x| -f64::sin(x));
    // let y = x.mapv(|x| (-(((x - 0.5) / 0.1).powi(2))).exp() / (0.1 * std::f64::consts::PI.sqrt()));
    // let dy = -2.0 * (&x - 0.5) * &y / (0.1f64.powi(2));

    // QTT decomposition of target function
    let mut yt = y.clone().into_dyn();
    let mut qtt: Vec<ArrayD<f64>> = Vec::with_capacity(depth);
    let mut rank: usize = 1;
    for k in 0..(depth - 1) {
        let yt_view = yt.to_shape((rank * 2, yt.len() / (rank * 2)))?;
        let (u, s, vt) = yt_view.thin_svd()?;
        let rank_new = s
            .iter()
            .position(|&x| x <= cutoff * s[0])
            .unwrap_or(s.len());
        let u = u.slice_move(s![.., 0..rank_new]);
        let s = s.slice_move(s![0..rank_new]);
        let vt = vt.slice_move(s![0..rank_new, ..]);
        let u = if k > 0 {
            u.to_shape((rank, 2, rank_new))?.into_owned().into_dyn()
        } else {
            u.into_dyn()
        };
        qtt.push(u);
        yt = Array2::from_diag(&s).dot(&vt).into_dyn();
        rank = rank_new;
    }
    let yt = yt.to_shape((rank, 2))?.into_owned().into_dyn();
    qtt.push(yt);
    print!("QTT virtual dimensions: [");
    for t in qtt.iter() {
        print!("{:?}, ", t.shape());
    }
    println!("]\n");

    // QTT for index shift operator
    let mut s_qtt: Vec<ArrayD<f64>> = Vec::with_capacity(depth);
    let mut s = Array3::<f64>::zeros((2, 2, 2));
    s[[0, 0, 0]] = 1.0;
    s[[1, 1, 0]] = 1.0;
    s[[1, 0, 1]] = 1.0;
    s[[0, 1, 1]] = 1.0;
    s_qtt.push(s.into_dyn());
    for _ in 1..(depth - 1) {
        let mut s = Array4::<f64>::zeros((2, 2, 2, 2));
        s[[0, 0, 0, 0]] = 1.0;
        s[[0, 1, 1, 0]] = 1.0;
        s[[0, 1, 0, 1]] = 1.0;
        s[[1, 0, 1, 1]] = 1.0;
        s_qtt.push(s.into_dyn());
    }
    let mut s = Array3::<f64>::zeros((2, 2, 2));
    s[[0, 1, 0]] = 1.0;
    s[[1, 0, 1]] = 1.0;
    s_qtt.push(s.into_dyn());
    print!("index shift operator virtual dimensions: [");
    for t in s_qtt.iter() {
        print!("{:?}, ", t.shape());
    }
    println!("]\n");

    // check for index shift operator
    if depth < 5 {
        let mut s_op = s_qtt[0].clone();
        for k in 1..(depth - 1) {
            s_op = einsum("ijk,klmn->iljmn", &[&s_op, &s_qtt[k]]).map_str_err()?;
            let shape = (
                s_op.shape()[0] * s_op.shape()[1],
                s_op.shape()[2] * s_op.shape()[3],
                s_op.shape()[4],
            );
            s_op = s_op.into_shape_clone(shape)?.into_dyn();
        }
        s_op = einsum("ijk,klm->iljm", &[&s_op, &s_qtt[depth - 1]]).map_str_err()?;
        let shape = (
            s_op.shape()[0] * s_op.shape()[1],
            s_op.shape()[2] * s_op.shape()[3],
        );
        s_op = s_op.into_shape_clone(shape)?.into_dyn();
        println!("index shift operator:\n{}\n", s_op);
    }

    // QTT for finite-difference operator
    let mut d_qtt: Vec<ArrayD<f64>> = Vec::with_capacity(depth);
    {
        let s = s_qtt[0].clone();
        let st = s.view().permuted_axes(vec![1, 0, 2]).into_owned();
        let mut d = Array3::<f64>::zeros((2, 2, 2 * s.shape()[2]));
        let w2 = s.shape()[2];
        d.slice_mut(ndarray::s![.., .., ..w2]).assign(&(-s));
        d.slice_mut(ndarray::s![.., .., w2..]).assign(&st);
        d_qtt.push(d.into_dyn());
    }
    for k in 1..(depth - 1) {
        let s = s_qtt[k].clone();
        let st = s.view().permuted_axes(vec![0, 2, 1, 3]).into_owned();
        let mut d = Array4::<f64>::zeros((2 * s.shape()[0], 2, 2, 2 * s.shape()[3]));
        let w0 = s.shape()[0];
        let w3 = s.shape()[3];

        d.slice_mut(ndarray::s![0..w0, .., .., 0..w3])
            .assign(&(2.0 * s));
        d.slice_mut(ndarray::s![w0.., .., .., w3..])
            .assign(&(2.0 * st));
        d_qtt.push(d.into_dyn());
    }
    {
        let s = s_qtt[depth - 1].clone();
        let st = s.view().permuted_axes(vec![0, 2, 1]).into_owned();
        let mut d = Array3::<f64>::zeros((2 * s.shape()[2], 2, 2));
        let w2 = s.shape()[2];
        d.slice_mut(ndarray::s![..w2, .., ..]).assign(&(2.0 * s));
        d.slice_mut(ndarray::s![w2.., .., ..]).assign(&(2.0 * st));
        d_qtt.push(d.into_dyn());
    }
    print!("finite-difference operator virtual dimensions: [");
    for t in d_qtt.iter() {
        print!("{:?}, ", t.shape());
    }
    println!("]\n");

    // check for finite-difference operator
    if depth < 5 {
        let mut d_op = d_qtt[0].clone();
        for k in 1..(depth - 1) {
            d_op = einsum("ijk,klmn->iljmn", &[&d_op, &d_qtt[k]]).map_str_err()?;
            let shape = (
                d_op.shape()[0] * d_op.shape()[1],
                d_op.shape()[2] * d_op.shape()[3],
                d_op.shape()[4],
            );
            d_op = d_op.into_shape_clone(shape)?.into_dyn();
        }
        d_op = einsum("ijk,klm->iljm", &[&d_op, &d_qtt[depth - 1]]).map_str_err()?;
        let shape = (
            d_op.shape()[0] * d_op.shape()[1],
            d_op.shape()[2] * d_op.shape()[3],
        );
        d_op = d_op.into_shape_clone(shape)?.into_dyn();
        println!("finite-difference operator:\n{}\n", d_op);
    }

    // apply finite-difference operator to target function
    let mut dy_qtt: Vec<ArrayD<f64>> = Vec::with_capacity(depth);
    {
        let t = einsum("jkl,kn->jnl", &[&d_qtt[0], &qtt[0]]).map_str_err()?;
        let shape = (t.shape()[0], t.shape()[1] * t.shape()[2]);
        dy_qtt.push(t.into_shape_clone(shape)?.into_dyn());
    }
    for k in 1..(depth - 1) {
        let tmp = einsum("ijkl,mkn->mijnl", &[&d_qtt[k], &qtt[k]]).map_str_err()?;
        let shape = (
            tmp.shape()[0] * tmp.shape()[1],
            tmp.shape()[2],
            tmp.shape()[3] * tmp.shape()[4],
        );
        dy_qtt.push(tmp.into_shape_clone(shape)?.into_dyn());
    }
    {
        let t = einsum("ijk,mk->mij", &[&d_qtt[depth - 1], &qtt[depth - 1]]).map_str_err()?;
        let shape = (t.shape()[0] * t.shape()[1], t.shape()[2]);
        dy_qtt.push(t.into_shape_clone(shape)?.into_dyn());
    }
    print!("derivative QTT virtual dimensions: [");
    for t in dy_qtt.iter() {
        print!("{:?}, ", t.shape());
    }
    println!("]\n");

    // reconstruction
    let mut dyr = dy_qtt[0].clone();
    for k in 1..(depth - 1) {
        let tmp = einsum("ij,jkl->ikl", &[&dyr, &dy_qtt[k]]).map_str_err()?;
        let shape = (tmp.shape()[0] * tmp.shape()[1], tmp.shape()[2]);
        dyr = tmp.into_shape_clone(shape)?.into_dyn();
    }
    let dyr = einsum("ij,jk->ik", &[&dyr, &dy_qtt[depth - 1]]).map_str_err()?;
    let dyr = dyr.flatten();

    // drop boundary points as derivative is not correct there due to boundary conditions
    let x = x.slice_move(ndarray::s![1..-2]); // keep 1..n-2 (python: 1:-2)
    let dy = dy.slice(ndarray::s![1..-2]);
    let dyr = dyr.slice_move(ndarray::s![1..-2]);

    if depth < 5 {
        println!("target derivative y' = {}\n", dy);
        println!("QTT derivative    y' = {}\n", dyr);
    }
    println!("error = {}\n", (&dyr - &dy).norm() / npoints as f64);

    // plots

    plot_target_vs_qtt(
        &BitMapBackend::new("target_vs_qtt_d.png", (1000, 600)).into_drawing_area(),
        &x,
        &dy,
        &dyr,
    )?;
    println!("Saved plot: {}", "target_vs_qtt_d.png");
    plot_error(
        &BitMapBackend::new("error_d.png", (1000, 600)).into_drawing_area(),
        &x,
        &dy,
        &dyr,
    )?;
    println!("Saved plot: {}", "error_d.png");

    Ok(())
}
