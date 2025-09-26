# 超初心者のためのテンソルネットワーク講座

* [CourseContents.md: 講義内容](CourseContents.md)
* [data: 実習で使用するデータファイル](data)
* 実習コード
    - 言語
        - [python: Python実習コード](python)
        - [julia: Julia実習コード](julia)
        - [rust: Rust実習コード](rust)
    - 実習コード
        - 1_svd
        - 2_image-compression
        - 3_tensor
        - 4_contraction
        - 5_mps2statevector
        - 6_statevector2mps
        - 7_tebd
        - 8_canonical-form (optional)
        - 9_function2qtt
        - 10_finite-difference
        - 11_fourier-transform (optional)

* materials: 参考資料
    - 以前の講義資料等

* Julia
    - 以下, `julia` ディレクトリ内での実行を想定
    - `julia`: `repl` 起動
        - 以下 `repl` 内で実行する
    - `using Pkg; Pkg.activate(".")`: プロジェクト環境を activate
        - `repl` を立ち上げるたびに実行する
        - `Pkg.status()` で <code>Status \`.../tn-basics/julia/Project.toml\`</code> のメッセージとともに `Project.toml` に記述されているパッケージが列挙されたら成功
        - `repl` 起動時に `julia --project` とオプションをつけることで、カレントディレクトリ内の `Project.toml` が自動的に検索され、プロジェクト環境が activate される (おすすめ)
        - `vscode` の `Julia` 拡張を利用していて、かつ、プロジェクト環境がうまく読み込まれない場合には、左下の `Julia env` をクリックし、`pick a folder` から `Project.toml` が配置されているディレクトリ (`tn-basics/julia`) を選択することで、ライブラリの入力補完やホバーによるドキュメント表示を利用することができる (`.vscode/settings.json` が作成されることに注意)
    - `Pkg.instantiate()`: 依存関係解決 + パッケージインストール
        - 初めてプログラムを実行する際にのみ実行する
        - `Manifest.toml` に依存関係が記述される
    - `include("path/to/<file>.jl")`: `<file>.jl` 実行
        - `julia --project <file>.jl` でも実行できるが、`repl` 内で `include` するとプログラムのコンパイル結果がキャッシュされるため、2回目以降の実行が高速になる
        - ファイルの内容を変更したとしても、再コンパイルはファイルごとに行われるため、`include` の方が効率的
