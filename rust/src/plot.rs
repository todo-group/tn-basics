use ndarray::{ArrayBase, Data, Ix1};
use plotters::{coord::Shift, prelude::*};

/// Plot y (target) and yr (qtt approximation) vs x
#[expect(clippy::missing_panics_doc, clippy::missing_errors_doc)]
pub fn plot_target_vs_qtt<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    x: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    y: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    yr: &ArrayBase<impl Data<Elem = f64>, Ix1>,
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

    let mut chart = ChartBuilder::on(area)
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
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], RED));

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
            .legend(|(x, y)| Circle::new((x, y), 3, BLUE));
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
            .legend(|(x, y)| Circle::new((x, y), 3, BLUE));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    // present (write file)
    area.present()?;
    Ok(())
}

/// Plot the error = y - yr vs x
#[expect(clippy::missing_panics_doc, clippy::missing_errors_doc)]
pub fn plot_error<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    x: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    y: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    yr: &ArrayBase<impl Data<Elem = f64>, Ix1>,
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

    let mut chart = ChartBuilder::on(area)
        .caption("Error", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, e_min..e_max)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .y_label_formatter(&|y| format!("{y:.2e}"))
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            x.iter().zip(errors.iter()).map(|(&xi, &yi)| (xi, yi)),
            &BLACK,
        ))?
        .label("error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], BLACK));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    area.present()?;

    Ok(())
}
