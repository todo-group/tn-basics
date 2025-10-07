use gtk4::{Application, Picture};
use gtk4::{ApplicationWindow, prelude::*};
use plotters::coord::Shift;
use plotters::prelude::*;
use plotters_gtk4::{Paintable, PaintableBackend};

pub fn plot_to_window<F>(draw_func: F, shape: (u32, u32))
where
    F: for<'a> Fn(&plotters::prelude::DrawingArea<PaintableBackend<'a>, Shift>),
{
    gtk4::init().unwrap();
    let app = Application::builder()
        .application_id("tn-basics.plotters-gtk4")
        .build();

    let area = Paintable::new(shape);
    let bd = PaintableBackend::new(&area);
    let root = bd.into_drawing_area();
    draw_func(&root);
    let picture = Picture::for_paintable(&area);

    app.connect_activate(move |app| {
        // We create the main window.
        let window = ApplicationWindow::builder()
            .application(app)
            .child(&picture)
            .title("GTK4 + Plotters Graph")
            .default_width(shape.0 as i32)
            .default_height(shape.1 as i32)
            .build();
        window.present();
    });
    app.run();
}
