use image::{GrayImage, ImageBuffer, ImageReader, Luma};
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2, s};
use plotters::{
    chart::ChartBuilder,
    coord::Shift,
    prelude::{
        BitMapBackend, DrawingArea, DrawingAreaErrorKind, DrawingBackend, IntoDrawingArea,
        IntoLogRange,
    },
    series::LineSeries,
    style::{BLUE, WHITE},
};
use tn_basics::EasySVD;

extern crate blas_src;

fn to_gray_ndarray<E>(img: &GrayImage, mut map_u8: impl FnMut(u8) -> E) -> Array2<E> {
    let (w, h) = img.dimensions();
    let arr_1d: Array1<_> = img
        .enumerate_pixels()
        .map(|(_, _, p)| map_u8(p.0[0]))
        .collect();
    arr_1d
        .into_shape_with_order((h as usize, w as usize))
        .unwrap()
}

fn to_gray_image<S: Data>(
    arr: &ArrayBase<S, Ix2>,
    mut map_e: impl FnMut(&S::Elem) -> u8,
) -> GrayImage {
    let h = arr.nrows();
    let w = arr.ncols();
    ImageBuffer::from_fn(w as u32, h as u32, |x, y| {
        Luma([map_e(&arr[(y as usize, x as usize)])])
    })
}

fn plot_singular_values<S: Data<Elem = f64>, DB: DrawingBackend>(
    s: &ArrayBase<S, Ix1>,
    area: &DrawingArea<DB, Shift>,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    area.fill(&WHITE)?;
    let max_y = s.iter().cloned().fold(f64::MIN, f64::max).max(1e-12);
    let min_y = s.iter().cloned().fold(f64::MAX, f64::min).min(1e+12);

    let mut chart = ChartBuilder::on(area)
        .caption("singular values", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0..(s.len() - 1), (min_y..max_y).log_scale())?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("index")
        .y_desc("λ_i")
        .draw()?;

    chart.draw_series(LineSeries::new((0..s.len()).map(|i| (i, s[i])), &BLUE))?;
    Ok(())
}

// Compress and reconstruct grayscale images using SVD

fn main() -> anyhow::Result<()> {
    let path = "../data/sqai-square-gray-rgb150ppi.jpg";

    // load image and convert to grayscale
    let image = ImageReader::open(path)?.decode()?.into_luma8();
    let array = to_gray_ndarray(&image, |v| v as f64);
    let [w, h] = array.shape() else {
        panic!("unexpected shape");
    };
    println!("image size; {} {}\n", h, w);
    to_gray_image(&array, |v| v.round().clamp(0.0, 255.0) as u8).save("original.png")?;
    println!("saved {}", "original.png");

    // SVD
    let (u, s, vt) = array.thin_svd()?;
    plot_singular_values(
        &s,
        &BitMapBackend::new("singular_values.png", (800, 500)).into_drawing_area(),
    )?;
    println!("saved {}", "singular_values.png");

    // image reconstruction with different ranks
    let ranks: [usize; _] = [1, 2, 4, 8, 16, 32, 64, 128, 256];
    for &r in &ranks {
        let r = r.min(s.len());
        let ur = u.slice(s![.., 0..r]);
        let sr = Array2::from_diag(&s.slice(s![0..r]));
        let vtr = vt.slice(s![0..r, ..]);
        let ar = ur.dot(&sr).dot(&vtr);
        let out = format!("reconstructed_rank_{}.png", r);
        to_gray_image(&ar, |v: &f64| v.round().clamp(0.0, 255.0) as u8).save(&out)?;
        println!("saved {}", out);
    }

    Ok(())
}
