use anyhow::Result;
use ndarray::{Array1, Array2, ArrayD, s};
use ndarray_einsum::einsum;
use ndarray_linalg::Norm;
use tn_basics::{EasySVD, MapStrToAnyhowErr};

fn main() -> Result<()> {
    let depth: usize = 4;
    let npoints: usize = 1 << depth;
    let cutoff: f64 = 1e-10;
    //let max_rank: usize = 4;

    // target function
    let x = Array1::linspace(0.0, 1.0, npoints);
    let y = x.mapv(f64::exp);
    //let y = x.mapv(f64::sin);
    //let y = x.mapv(|t| ( -(((t-0.5)/0.1).powi(2)) ).exp() / (0.1 * PI.sqrt()) );

    if depth < 5 {
        println!("x = {}", x);
        println!("y = {}\n", y);
    }

    // QTT decomposition
    let mut yt = y.clone().into_dyn();
    let mut qtt: Vec<ArrayD<f64>> = Vec::new();
    let mut rank: usize = 1;

    for k in 0..(depth - 1) {
        println!("depth: {}", k);
        let v_view = yt.to_shape((rank * 2, yt.len() / (rank * 2)))?;
        let (u, s, vt) = v_view.thin_svd().unwrap();
        println!("singular values: {}", s);
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
    println!("depth: {}", depth - 1);
    let yt = yt.to_shape((rank, 2))?.into_owned().into_dyn();
    qtt.push(yt);
    if depth < 5 {
        println!("qtt:");
        for (i, t) in qtt.iter().enumerate() {
            println!("  {}:\n{}", i, t);
        }
    }

    // reconstruction
    let mut yr = qtt[0].clone();
    for k in 1..(depth - 1) {
        let tmp = einsum("ij,jkl->ikl", &[&yr, &qtt[k]]).map_str_err()?;
        let shape = (tmp.shape()[0] * tmp.shape()[1], tmp.shape()[2]);
        yr = tmp.into_shape_clone(shape)?.into_dyn();
    }
    let yr = einsum("ij,jk->ik", &[&yr, &qtt[depth - 1]]).map_str_err()?;
    let yr = yr.flatten();
    if depth < 5 {
        println!("reconstructed y:");
        for (i, t) in qtt.iter().enumerate() {
            println!("  {}:\n{}", i, t);
        }
    }

    println!("error = {}", (&yr - &y).norm() / npoints as f64);

    plot_target_vs_qtt(
        &BitMapBackend::new("target_vs_qtt.png", (800, 600)).into_drawing_area(),
        x.as_slice().unwrap(),
        y.as_slice().unwrap(),
        yr.as_slice().unwrap(),
    )?;
    plot_error(
        &BitMapBackend::new("error.png", (800, 600)).into_drawing_area(),
        x.as_slice().unwrap(),
        y.as_slice().unwrap(),
        yr.as_slice().unwrap(),
    )?;

    Ok(())
}

use plotters::{coord::Shift, prelude::*};

/// Plot y (target) and yr (qtt approximation) vs x
fn plot_target_vs_qtt<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    x: &[f64],
    y: &[f64],
    yr: &[f64],
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    area.fill(&WHITE)?;

    // Determine x-range and y-range
    let x_min = *x.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let x_max = *x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let y_min = *y
        .iter()
        .chain(yr.iter())
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let y_max = *y
        .iter()
        .chain(yr.iter())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let mut chart = ChartBuilder::on(&area)
        .caption("Target vs QTT", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh().disable_mesh().draw()?;

    // draw the target curve (continuous line)
    chart
        .draw_series(LineSeries::new(
            x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi, yi)),
            &RED,
        ))?
        .label("target")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &RED));

    // draw QTT points (either full or subsampled)
    let npoints = x.len();
    if npoints <= 32 {
        chart
            .draw_series(
                x.iter()
                    .zip(yr.iter())
                    .map(|(xi, yi)| Circle::new((*xi, *yi), 3, ShapeStyle::from(&BLUE).filled())),
            )?
            .label("QTT")
            .legend(|(x, y)| Circle::new((x, y), 3, &BLUE));
    } else {
        let stride = npoints / 32;
        chart
            .draw_series(
                x.iter()
                    .zip(yr.iter())
                    .enumerate()
                    .filter_map(|(i, (xi, yi))| {
                        if i % stride == 0 {
                            Some(Circle::new((*xi, *yi), 3, ShapeStyle::from(&BLUE).filled()))
                        } else {
                            None
                        }
                    }),
            )?
            .label("QTT")
            .legend(|(x, y)| Circle::new((x, y), 3, &BLUE));
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    // present (write file)
    Ok(())
}

/// Plot the error = y - yr vs x
fn plot_error<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    x: &[f64],
    y: &[f64],
    yr: &[f64],
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    area.fill(&WHITE)?;

    // Compute error
    let errors: Vec<f64> = y
        .iter()
        .zip(yr.iter())
        .map(|(&yi, &yri)| yi - yri)
        .collect();

    // Determine ranges
    let x_min = *x.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let x_max = *x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let e_min = *errors
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let e_max = *errors
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let mut chart = ChartBuilder::on(&area)
        .caption("Error", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, e_min..e_max)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .y_label_formatter(&|y| format!("{:.2e}", y))
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            x.iter().zip(errors.iter()).map(|(&xi, &yi)| (xi, yi)),
            &BLACK,
        ))?
        .label("error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLACK));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}
