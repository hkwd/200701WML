
"""
WML Decision Optimization モデル - Diet Problem (食事問題)

このスクリプトは、Watson Machine Learning (WML) 上で実行される
Decision Optimizationモデルのメインファイルです。
栄養バランスを保ちながら食事コストを最小化する最適化問題を解きます。
"""

from docplex.mp.model import Model
from docplex.mp.progress import SolutionListener
from docplex.util.environment import get_environment
import pandas
from os.path import join
import json
import zipfile
import io

# ============================================================================
# 定数定義
# ============================================================================

INPUT_ZIP = "input.zip"
OUTPUT_ZIP = "output.zip"
MODEL_SCHEMA = "model_schema.json"
INPUT_FILE_NAMES = ['diet_food', 'diet_nutrients', 'diet_food_nutrients']


# ============================================================================
# ヘルパー関数
# ============================================================================

def get_zip_path(directory=None):
    """
    input.zipのパスを構築する

    Args:
        directory: ディレクトリパス

    Returns:
        str: input.zipのフルパス
    """
    return join(directory, INPUT_ZIP) if directory else INPUT_ZIP


# ============================================================================
# データ入力関数
# ============================================================================

def load_dtype_schemas(directory=None):
    """
    input.zip内のmodel_schema.jsonからデータ型情報を読み込む

    Args:
        directory: ディレクトリパス

    Returns:
        dict: {ファイル名: {カラム名: データ型}} の辞書
    """
    dtype_schemas = {}
    zip_path = get_zip_path(directory)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            with zf.open(MODEL_SCHEMA) as f:
                input_schemas = json.load(f)

                if 'input' in input_schemas:
                    for input_schema in input_schemas['input']:
                        dtype_schema = {}

                        if 'fields' in input_schema:
                            for input_schema_field in input_schema['fields']:
                                dtype_schema[input_schema_field['name']] = input_schema_field['type']

                            if len(dtype_schema) > 0:
                                dtype_schemas[input_schema['id']] = dtype_schema
    except (KeyError, FileNotFoundError) as e:
        print(f"Warning: Could not load schema from {zip_path}: {e}")

    print("Loaded dtype schemas:", dtype_schemas)
    return dtype_schemas


def read_csv_from_zip(filename, directory=None, dtype_schema=None):
    """
    input.zip内のCSVファイルをpandas DataFrameとして読み込む

    Args:
        filename: ファイル名（拡張子なし）
        directory: ディレクトリパス
        dtype_schema: データ型スキーマ

    Returns:
        pandas.DataFrame: 読み込まれたデータフレーム
    """
    csv_file = f"{filename}.csv"
    zip_path = get_zip_path(directory)

    # パラメータ設定
    params = {'encoding': 'utf8'}
    if dtype_schema:
        params['dtype'] = dtype_schema

    # input.zipから読み込む
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open(csv_file) as f:
            df = pandas.read_csv(io.BytesIO(f.read()), **params)

    return df


def load_all_inputs(directory=None):
    """
    全ての入力CSVファイルを読み込む

    Args:
        directory: CSVファイルが格納されているディレクトリパス

    Returns:
        dict: {ファイル名: DataFrame} の辞書
    """
    # スキーマ情報を読み込む
    dtype_schemas = load_dtype_schemas(directory)

    # 各CSVファイルを読み込む
    inputs = {}

    for filename in INPUT_FILE_NAMES:
        csv_filename = f"{filename}.csv"
        dtype_schema = dtype_schemas.get(csv_filename)
        inputs[filename] = read_csv_from_zip(filename, directory, dtype_schema)

    return inputs


def write_all_outputs(outputs):
    """
    全ての出力DataFrameをCSVファイルとして書き出す
    個別のCSVファイルに加えて、output.zipも作成する

    WML環境の出力ストリームを使用して、最適化結果を
    CSVファイルとして保存します。これらのファイルは
    WMLのジョブ結果として取得できます。

    Args:
        outputs: 出力データの辞書 {ファイル名: DataFrame}

    Example:
        >>> outputs = {
        ...     'solution': solution_df,
        ...     'kpis': kpis_df
        ... }
        >>> write_all_outputs(outputs)
    """
    # 個別のCSVファイルを書き出す
    for name, df in outputs.items():
        csv_file = f'{name}.csv'
        print(csv_file)

        # WML環境の出力ストリームを使用してCSVを書き出す (Python 3.x)
        with get_environment().get_output_stream(csv_file) as fp:
            fp.write(df.to_csv(index=False).encode(encoding='utf8'))

    # output.zipを作成
    if len(outputs) > 0:
        print(f"Creating {OUTPUT_ZIP}")
        with get_environment().get_output_stream(OUTPUT_ZIP) as zip_fp:
            # メモリ上にZIPファイルを作成
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # 各DataFrameをCSVとしてZIPに追加
                for name, df in outputs.items():
                    csv_file = f'{name}.csv'
                    csv_content = df.to_csv(index=False)
                    zipf.writestr(csv_file, csv_content)

            # ZIPファイルの内容を出力ストリームに書き込む
            zip_fp.write(zip_buffer.getvalue())
        print(f"{OUTPUT_ZIP} created successfully")
    else:
        print("Warning: no outputs written")


# ============================================================================
# ソリューション構築とリスナー
# ============================================================================

def build_solution(sol):
    """
    最適化ソリューションからDataFrameを構築し、CSVとして出力

    Args:
        sol: DOcplexのソリューションオブジェクト
    """
    # ソリューションをDataFrameに変換
    solution_df = pandas.DataFrame(columns=['Food', 'value'])

    for index, dvar in enumerate(sol.iter_variables()):
        solution_df.loc[index, 'Food'] = dvar.to_string()
        solution_df.loc[index, 'value'] = dvar.solution_value

    outputs = {}
    outputs['solution'] = solution_df

    # 出力ファイルを生成
    write_all_outputs(outputs)


class SolutionKeeper(SolutionListener):
    """
    中間ソリューションを追跡する特殊なSolutionListenerの実装

    最適化の途中で見つかった解を逐次保存し、WMLで中間結果を
    確認できるようにします。
    """

    def __init__(self):
        """初期化"""
        SolutionListener.__init__(self)
        self.index = -1

    def notify_solution(self, sol):
        """
        新しいソリューションが見つかった時に呼ばれるコールバック

        Args:
            sol: 見つかったソリューション
        """
        self.index += 1
        # 中間ソリューションを保存
        build_solution(sol)


# ============================================================================
# メイン処理: データ読み込みとモデル構築
# ============================================================================

# 全ての入力CSVファイルを読み込む
inputs = load_all_inputs()

# 各データファイルを取得
food = inputs['diet_food']                    # 食品データ（名前、コスト、数量制限）
nutrients = inputs['diet_nutrients']          # 栄養素データ（名前、必要量の範囲）
food_nutrients = inputs['diet_food_nutrients']  # 食品ごとの栄養素含有量
food_nutrients.set_index('Food', inplace=True)  # 食品名をインデックスに設定


# ============================================================================
# 最適化モデルの構築
# ============================================================================

# モデルの作成
mdl = Model(name='diet')

# 中間ソリューションのテーブルを取得するには、SolutionListenerを追加する必要があります
# これにより、ソリューションテーブルを作成して公開できます
mdl.add_progress_listener(SolutionKeeper())

# 決定変数の作成: Food.qmin以上、Food.qmax以下に制限
# 各食品の購入量を表す連続変数を作成
qty = food[['name', 'qmin', 'qmax']].copy()
qty['var'] = qty.apply(lambda x: mdl.continuous_var(lb=x['qmin'],
                                                    ub=x['qmax'],
                                                    name=x['name']),
                       axis=1)
# 名前をインデックスにする
qty.set_index('name', inplace=True)

# 栄養素の範囲を制限し、KPIとしてマーク
# 各栄養素について、必要量の範囲内に収まるように制約を追加
for n in nutrients.itertuples():
    # 各食品から摂取する栄養素の合計を計算
    amount = mdl.sum(qty.loc[f.name]['var'] * food_nutrients.loc[f.name][n.name]
                     for f in food.itertuples())
    # 栄養素の最小値と最大値の範囲制約を追加
    mdl.add_range(n.qmin, amount, n.qmax)
    # KPI（重要業績評価指標）として登録
    mdl.add_kpi(amount, publish_name='Total %s' % n.name)

# 目的関数: コストを最小化
# 各食品の購入量 × 単価の合計を最小化
obj = mdl.sum(qty.loc[f.name]['var'] * f.unit_cost for f in food.itertuples())
mdl.add_kpi(obj, publish_name="Minimal cost")
mdl.minimize(obj)

# モデル情報を表示
mdl.print_information()

# ============================================================================
# 最適化の実行
# ============================================================================

# モデルを解く
ok = mdl.solve()

# ソリューションを表示
mdl.print_solution()

# 最終ソリューションをCSVファイルとして出力
build_solution(mdl.solution)
