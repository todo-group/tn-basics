use ndarray::prelude::*;
use image::{ImageReader, DynamicImage, GenericImageView, ImageBuffer, Luma};
use plotters::prelude::*;
use tn_basics::ThinSVD;

extern crate blas_src;

fn to_ndarray_gray_u8(img: &DynamicImage) -> Array2<u8> {
    // グレイスケール化（Luma8）
    let g = img.to_luma8();
    let (w, h) = g.dimensions();
    let mut arr = Array2::<u8>::zeros((h as usize, w as usize));
    for (x, y, p) in g.enumerate_pixels() {
        arr[(y as usize, x as usize)] = p.0[0];
    }
    arr
}

fn save_gray_u8(path: &str, a: &Array2<f64>) -> anyhow::Result<()> {
    // 0..255 にクリップして u8 に落とす
    let h = a.nrows();
    let w = a.ncols();
    let mut buf: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let v = a[(y, x)].round().clamp(0.0, 255.0) as u8;
            buf.put_pixel(x as u32, y as u32, Luma([v]));
        }
    }
    buf.save(path)?;
    Ok(())
}

fn plot_singular_values(path: &str, s: &Array1<f64>) -> anyhow::Result<()> {
    let root = BitMapBackend::new(path, (800, 500)).into_drawing_area();
    root.fill(&WHITE)?;
    let max_y = s.iter().cloned().fold(f64::MIN, f64::max).max(1e-12);

    let min_y = s.iter().cloned().fold(f64::MAX, f64::min).min(1e10);

    let mut chart = ChartBuilder::on(&root)
        .caption("singular values", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0..(s.len() - 1), (min_y..max_y).log_scale())?;

    chart.configure_mesh().disable_mesh()
        .x_desc("index")
        .y_desc("λ_i")
        .draw()?;

    chart.draw_series(LineSeries::new(
        (0..s.len()).map(|i| (i, s[i])),
        &BLUE,
    ))?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let path = "../data/sqai-square-gray-rgb150ppi.jpg";

    // 画像読み込み
    let img = ImageReader::open(path)?.decode()?;
    let (w, h) = img.dimensions();
    println!("image size; {} {}\n", h, w);

    // ndarray<f64> へ（0..255）
    let arr_u8 = to_ndarray_gray_u8(&img);
    let a: Array2<f64> = arr_u8.map(|&v| v as f64);

    // 元画像を保存（確認用）
    save_gray_u8("original.png", &a)?;

    // SVD（thin）
    let (u, s, vt) = a.thin_svd(true, true)?;
    let u = u.unwrap();
    let vt = vt.unwrap();

    // 特異値プロット（対数軸）
    plot_singular_values("singular_values.png", &s)?;

    // ランクごとの再構成
    let ranks: [usize; _] = [1, 2, 4, 8, 16, 32, 64, 128, 256];
    for &r in &ranks {
        let r = r.min(s.len());
        let ur = u.slice(s![.., 0..r]).to_owned();
        let sr = Array2::from_diag(&s.slice(s![0..r]).to_owned());
        let vtr = vt.slice(s![0..r, ..]).to_owned();
        let ar = ur.dot(&sr).dot(&vtr);

        let out = format!("reconstructed_rank_{}.png", r);
        save_gray_u8(&out, &ar)?;
        println!("saved {}", out);
    }

    Ok(())
}
