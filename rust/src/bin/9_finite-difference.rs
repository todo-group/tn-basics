use anyhow::Result;
use ndarray::{Array1, Array2, Array3, Array4, Axis};
use ndarray_einsum_beta::einsum;
use ndarray_linalg::svd::SVD;
use plotters::prelude::*;

fn main() -> Result<()> {
    let depth: usize = 4;
    let npoints: usize = 1 << depth;
    let cutoff: f64 = 1e-10;
    let max_rank: usize = 4;

    // target function & derivative
    let x: Array1<f64> = Array1::linspace(0.0, 1.0, npoints);
    let y: Array1<f64> = x.mapv(|t| t.exp());
    let dy: Array1<f64> = x.mapv(|t| t.exp());
    // let y = x.mapv(|t| t.cos()); let dy = x.mapv(|t| -t.sin());

    // QTT decomposition of y
    let mut yt = y.clone();
    let mut qtt: Vec<Array3<f64>> = Vec::with_capacity(depth);
    let mut rank: usize = 1;
    for _k in 0..(depth - 1) {
        let cols = yt.len() / (rank * 2);
        let ymat = yt.clone().into_shape((rank * 2, cols))?;
        let (u_opt, s, vt_opt) = ymat.svd(true, true)?;
        let (u, vt) = (u_opt.unwrap(), vt_opt.unwrap());
        // cutoff + rank cap
        let thr = cutoff * s[0];
        let mut r = 0usize;
        for &sv in s.iter() { if sv > thr { r += 1; } else { break; } if r == max_rank { break; } }
        let mut ukeep = u.slice(ndarray::s![.., 0..r]).to_owned();
        let vkeep = vt.slice(ndarray::s![0..r, ..]).to_owned();
        ukeep = ukeep.into_shape((rank, 2, r))?;
        qtt.push(ukeep);
        // next right core
        let mut diag = Array2::<f64>::zeros((r, r));
        for i in 0..r { diag[(i,i)] = s[i]; }
        yt = (diag.dot(&vkeep)).into_raw_vec().into();
        rank = r;
    }
    qtt.push(yt.into_shape((rank, 2, 1))?);
    println!("QTT virtual dimensions: {:?}\n", qtt.iter().map(|a| a.dim()).collect::<Vec<_>>());

    // QTT shift operator cores
    let mut s_qtt: Vec<Array4<f64>> = Vec::with_capacity(depth);
    // first
    let mut s0 = Array4::<f64>::zeros((1,2,2,2));
    s0[[0,0,0,0]] = 1.0; s0[[0,1,1,0]] = 1.0; s0[[0,1,0,1]] = 1.0; s0[[0,0,1,1]] = 1.0;
    s_qtt.push(s0);
    // middle
    for _ in 1..(depth-1) {
        let mut s = Array4::<f64>::zeros((2,2,2,2));
        s[[0,0,0,0]] = 1.0; s[[0,1,1,0]] = 1.0; s[[0,1,0,1]] = 1.0; s[[1,0,1,1]] = 1.0;
        s_qtt.push(s);
    }
    // last
    let mut sl = Array4::<f64>::zeros((2,2,2,1));
    sl[[0,1,0,0]] = 1.0; sl[[1,0,1,0]] = 1.0;
    s_qtt.push(sl);
    println!("index shift operator virtual dimensions: {:?}\n",
        s_qtt.iter().map(|a| a.dim()).collect::<Vec<_>>());

    // finite-difference operator cores
    let mut d_qtt: Vec<Array4<f64>> = Vec::with_capacity(depth);
    // first
    {
        let s = s_qtt[0].clone();
        let st = s.view().permuted_axes([0,2,1,3]).to_owned();
        let mut d = Array4::<f64>::zeros((s.dim().0, 2,2, 2*s.dim().3));
        // left block: +s, right block: -st
        let w = s.dim().3;
        d.slice_mut(ndarray::s![.., .., .., 0..w]).assign(&s);
        d.slice_mut(ndarray::s![.., .., .., w..2*w]).assign(&(-&st));
        d_qtt.push(d);
    }
    // middle
    for k in 1..(depth-1) {
        let s = s_qtt[k].clone();
        let st = s.view().permuted_axes([0,2,1,3]).to_owned();
        let mut d = Array4::<f64>::zeros((2*s.dim().0, 2,2, 2*s.dim().3));
        let (a,w) = (s.dim().0, s.dim().3);
        d.slice_mut(ndarray::s![0..a, .., .., 0..w]).assign(&(2.0*&s));
        d.slice_mut(ndarray::s![a..2*a, .., .., w..2*w]).assign(&(2.0*&st));
        d_qtt.push(d);
    }
    // last
    {
        let s = s_qtt[depth-1].clone();
        let st = s.view().permuted_axes([0,2,1,3]).to_owned();
        let mut d = Array4::<f64>::zeros((2*s.dim().0, 2,2, s.dim().3));
        let a = s.dim().0;
        d.slice_mut(ndarray::s![0..a, .., .., ..]).assign(&(2.0*&s));
        d.slice_mut(ndarray::s![a..2*a, .., .., ..]).assign(&(2.0*&st));
        d_qtt.push(d);
    }
    println!("finite-difference operator virtual dimensions: {:?}\n",
        d_qtt.iter().map(|a| a.dim()).collect::<Vec<_>>());

    // apply D to QTT(y): t = einsum("ijkl,mjn->miknl") → reshape (mi, k, nl)
    let mut dy_qtt: Vec<Array3<f64>> = Vec::with_capacity(depth);
    for k in 0..depth {
        let d = &d_qtt[k]; let yk = &qtt[k];
        let tmp: ndarray::Array5<f64> = einsum("ijkl,mjn->miknl", &[d, yk]).unwrap();
        let (m,i,kdim,n,l) = (tmp.dim().0, tmp.dim().1, tmp.dim().2, tmp.dim().3, tmp.dim().4);
        dy_qtt.push(tmp.into_shape((m*i, kdim, n*l))?);
    }
    println!("derivative QTT virtual dimensions: {:?}\n",
        dy_qtt.iter().map(|a| a.dim()).collect::<Vec<_>>());

    // reconstruction
    let mut dyr = dy_qtt[0].clone().into_shape((dy_qtt[0].dim().0 * dy_qtt[0].dim().1, dy_qtt[0].dim().2))?.to_owned();
    for k in 1..depth {
        let tmp: ndarray::Array3<f64> = einsum("ij,jkl->ikl", &[&dyr, &dy_qtt[k]]).unwrap();
        let (a,b,c) = tmp.dim();
        dyr = tmp.into_shape((a*b, c))?;
    }
    let mut dyr_vec = dyr.into_raw_vec();

    // drop edges
    let x2  = x.slice(ndarray::s![1..-1]).to_owned(); // keep 1..n-2 (python: 1:-2)
    let dy2 = dy.slice(ndarray::s![1..-1]).to_owned();
    dyr_vec = dyr_vec[1..(dyr_vec.len()-1)].to_vec();

    // error
    let err = x2.iter().enumerate().map(|(i,_)| {
        let d = dy2[i] - dyr_vec[i];
        d*d
    }).sum::<f64>().sqrt() / (x2.len() as f64);
    println!("target derivative y' = {:?}\n", dy2);
    println!("QTT derivative   y' = {:?}\n", dyr_vec);
    println!("error = {}\n", err);

    // plots
    std::fs::create_dir_all("plots")?;
    let path1 = "plots/qtt_fd.png";
    let root = BitMapBackend::new(path1, (1000, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let ymin = dy2.iter().cloned().fold(f64::INFINITY, f64::min);
    let ymax = dy2.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption("QTT finite-difference", ("sans-serif", 30))
        .set_label_area_size(LabelAreaPosition::Left, 50)
        .set_label_area_size(LabelAreaPosition::Bottom, 50)
        .build_cartesian_2d(0.0..1.0, ymin..ymax)?;
    chart.configure_mesh().x_desc("x").y_desc("y'").draw()?;
    chart.draw_series(LineSeries::new(
        x2.iter().zip(dy2.iter()).map(|(xi, yi)| (*xi, *yi)),
        &BLACK,
    ))?.label("target").legend(|(x,y)| PathElement::new(vec![(x,y),(x+20.0,y)], &BLACK));
    chart.draw_series(x2.iter().zip(dyr_vec.iter()).map(|(xi, yi)| {
        Circle::new((*xi, *yi), 3, BLUE.filled())
    }))?.label("QTT 1").legend(|(x,y)| Circle::new((x,y), 3, BLUE.filled()));
    chart.configure_series_labels().border_style(&BLACK).draw()?;
    println!("Saved plot: {}", path1);

    // error plot
    let path2 = "plots/qtt_fd_err.png";
    let root2 = BitMapBackend::new(path2, (1000, 400)).into_drawing_area();
    root2.fill(&WHITE)?;
    let mut chart2 = ChartBuilder::on(&root2)
        .margin(20).caption("error", ("sans-serif", 24))
        .set_label_area_size(LabelAreaPosition::Left, 50)
        .set_label_area_size(LabelAreaPosition::Bottom, 50)
        .build_cartesian_2d(0.0..1.0, -0.1..0.1)?; // y軸は適宜
    chart2.configure_mesh().x_desc("x").y_desc("error").draw()?;
    chart2.draw_series(LineSeries::new(
        x2.iter().zip(dy2.iter().zip(dyr_vec.iter())).map(|(xi,(t,q))| (*xi, t - q)),
        &RED,
    ))?.label("error 1").legend(|(x,y)| PathElement::new(vec![(x,y),(x+20.0,y)], &RED));
    chart2.configure_series_labels().border_style(&BLACK).draw()?;
    println!("Saved plot: {}", path2);

    Ok(())
}
