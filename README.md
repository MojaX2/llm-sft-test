# 動画キャプショニング: Reasoning VLM を sm-swift で SFT する実装ガイド

本READMEは日本語で記述し、モデル出力（`thinking` と `caption`）は英語で統一します。

このリポジトリでは、次の 3 ステップで実験を進めることを想定します。

1. 強いモデル（GPT-5.4）で、動画ごとの `thinking` と `caption` を生成して教師データ化する
2. sm-swift で reasoning VLM（`Qwen/Qwen3-VL-2B-Instruct`）を SFT する
3. 学習済みモデルと比較対象モデルのキャプションを、強いモデル（GPT-5.4）で比較評価する

以下は「そのまま実装しやすい」粒度での設計です。

---

## 1. プロジェクト構成（推奨）

```text
.
├── input_videos/                    # 元動画
├── output_frames/                   # 既存のフレーム抽出結果
├── data/
│   ├── metadata/
│   │   ├── video_manifest.csv       # 動画の一覧（実験対象フラグ込み）
│   │   └── active_videos.txt        # 現在回すvideo_id一覧
│   ├── ground_truth/
│   │   ├── raw/                     # GPT-5.4の生出力
│   │   └── final/                   # 人手レビュー後の確定正解
│   ├── sft/                         # sm-swift学習データ
│   ├── predictions/                 # 各モデル推論結果
│   └── eval/                        # GPT-5.4比較評価結果
├── scripts/
│   ├── 01_build_teacher_data.py     # 動画→thinking/caption生成
│   ├── 02_make_swift_dataset.py     # sm-swift形式に変換
│   ├── 03_infer_captions.py         # 各モデルでキャプション生成
│   └── 04_eval_pairwise.py          # GPTでペア比較評価
├── extract_frames_sample.py
└── README.md
```

### 1.1 input_videos の対象動画リスト

現在の対象は `clerk.mp4` です。実験で使う一覧は `data/metadata/video_manifest.csv` で管理します。

`data/metadata/video_manifest.csv`:

```csv
video_id,video_path,scenario,viewpoint,primary_actor,other_actor,language_target,split,active
clerk_001,input_videos/clerk.mp4,cafe_simulation,first_person,clerk,customer,en,train,true
```

補助的に、現在有効なIDのみを `data/metadata/active_videos.txt` に保存します。

```text
clerk_001
```

### 1.2 正解データ保存構造（推奨）

- `data/ground_truth/raw/`
	- GPT-5.4 が最初に生成した生データ（再生成・差分比較用）
	- 例: `data/ground_truth/raw/clerk_001.json`
- `data/ground_truth/final/`
	- 品質確認後に確定した正解データ（学習投入用）
	- 例: `data/ground_truth/final/clerk_001.json`

保存ポリシー:

- まず `raw` に保存
- ルールチェック（英語、意図記述、観察ベース）後に `final` へ昇格
- 学習データ化は `final` のみを入力として使う

---

## 2. ステップ1: 強いモデルで `thinking` + `caption` を生成

### 2.0 このプロジェクトのタスク定義（固定）

- 強いモデル（教師データ生成・評価）: GPT-5.4
- 訓練対象モデル: Qwen/Qwen3-VL-2B-Instruct

- 環境: シミュレーション環境
- カメラ視点: 一人称視点（POV）
- 話者/主体: カフェ店員（カメラ装着者）
- 他の登場人物: お客さん

出力方針:

- `thinking`（英語）:
	- 動画に実際に写っている「人・物体・動き」を観察ベースで記述
	- 推測は最小限にし、根拠のない断定を避ける
- `caption`（英語）:
	- 店員（first-person actor）の意図
	- お客さん（other actor）の意図
	- 上記2者の意図を1つの自然な説明文としてまとめる

### 2.1 入力データの単位

- 最低限: `video_id`, `video_path`
- 可能なら追加: `domain`, `language`, `duration_sec`

`data/metadata/video_manifest.csv` 例:

```csv
video_id,video_path,scenario,viewpoint,primary_actor,other_actor,language_target,split,active
clerk_001,input_videos/clerk.mp4,cafe_simulation,first_person,clerk,customer,en,train,true
```

### 2.2 出力スキーマ（teacher_raw）

`data/ground_truth/raw/teacher_raw.jsonl` の 1 行 1 サンプル:

```json
{
	"video_id": "clerk_001",
	"video_path": "input_videos/clerk.mp4",
	"thinking": "The camera wearer reaches toward a cup, places it near the register, and extends a hand toward the customer. The customer moves closer, presents an item, and waits facing the counter.",
	"caption": "The clerk appears to be completing checkout and handing over the prepared item, while the customer intends to receive the order and finalize payment.",
	"language": "en"
}
```

### 2.3 生成時のプロンプト方針

- `thinking` は英語で「観察根拠が残る短い推論」に統一
- `caption` は英語で、店員と客の意図を両方含む1文
- 禁止事項を明示:
	- 映っていない情報の断定
	- 個人属性の過度な推測
	- 差別的表現

推奨システム指示（要約）:

- You are generating training data for first-person cafe-clerk video understanding.
- The camera wearer is the clerk; other visible people are customers.
- Return JSON only with keys: `thinking`, `caption`.
- `thinking` must be in English, 1-3 sentences, and only describe visible people/objects/actions.
- `caption` must be in English, 1 sentence, and explain both clerk intent and customer intent.
- If intention is uncertain, use cautious wording such as "appears to" or "seems to".

### 2.4 実装上の注意

- APIの再実行に備えて `video_id` 単位のキャッシュを保存
- レート制限対応として指数バックオフを実装
- 同じ動画で複数回生成し、最良候補を選別する運用も可能

### 2.5 API 入出力仕様（固定）

- GPT-5.4（OpenAI API）:
	- 動画はフレーム列に分解して送る
	- 各フレーム画像を base64 エンコードして `input_image` として投入する
	- 例: `frame_000.jpg`, `frame_001.jpg` ... を時系列順に並べて同一リクエストへ含める
- Qwen/Qwen3-VL-2B-Instruct（推論時）:
	- 動画ファイル全体を base64 エンコードして送る
	- 画像フレーム分割ではなく、動画1本を1つのペイロードとして扱う

実装ポリシー:

- 教師データ生成（Step1）と最終評価（Step3）の GPT-5.4 呼び出しは「フレーム base64」方式を使う
- Qwen3-VL の推論呼び出しは「動画 base64」方式を使う

---

## 3. ステップ2: sm-swift で SFT

### 3.1 学習形式

reasoning VLM では、次のどちらかに揃えるのが扱いやすいです。

1. `assistant` 出力に `thinking` と `caption` をまとめる
2. `assistant` は `caption` のみ、`thinking` は補助情報として別キー保持

まずは 1 で開始し、推論時に `caption` のみ抽出する方式が実装コスト低めです。

`data/sft_train.jsonl` 例（会話形式）:

```json
{
	"messages": [
		{
			"role": "user",
			"content": "This is a first-person cafe clerk video. Write English thinking from visible actions, then provide one English caption explaining both clerk intent and customer intent."
		},
		{
			"role": "assistant",
			"content": "<thinking>The camera wearer places a drink on the counter and reaches toward the payment area. The customer steps forward, presents a card, and watches the clerk's hand movement.</thinking>\n<caption>The clerk appears to be finishing the transaction and preparing to hand over the order, while the customer intends to pay and collect the item.</caption>"
		}
	],
	"videos": ["input_videos/clerk.mp4"],
	"id": "clerk_001"
}
```

### 3.2 学習実行（例）

実コマンドは利用モデル・環境で調整してください。考え方は以下です。

```bash
swift sft \
	--model Qwen/Qwen3-VL-2B-Instruct \
	--train_type lora \
	--dataset data/sft_train.jsonl \
	--val_dataset data/sft_valid.jsonl \
	--output_dir outputs/qwen3vl2b_sft \
	--num_train_epochs 3 \
	--per_device_train_batch_size 1 \
	--gradient_accumulation_steps 8 \
	--learning_rate 1e-4 \
	--max_length 4096 \
	--logging_steps 10 \
	--save_steps 200
```

ポイント:

- 最初は LoRA で軽量に回す
- 動画入力が重いので、バッチ小さめ + accumulation で調整
- train/valid は動画のシーンが重ならないよう分割

### 3.3 Qwen3-VL 推論I/O仕様（固定）

- `scripts/03_infer_captions.py` で Qwen3-VL を呼ぶ際は、`video_path` のバイナリを base64 化して送信する
- 同一動画に対して、推論時プロンプトは学習時と同等の制約を維持する
	- thinking: visible actions only, English
	- caption: one English sentence with both clerk/customer intent

---

## 4. ステップ3: 学習済み vs 比較対象 の GPT 評価

### 4.1 比較対象

- `base`: 未学習モデル（または別ベースライン）
- `ft`: SFT後モデル
- 必要に応じて `gpt54_direct`: GPT-5.4 直接キャプション

### 4.2 推論結果の形式

`data/pred_base.jsonl`, `data/pred_ft.jsonl`:

```json
{
	"video_id": "clerk_001",
	"caption": "The clerk is working near the register while preparing to complete a customer order."
}
```

### 4.3 GPT でのペアワイズ評価

評価入力:

- 動画フレーム列（OpenAI API に base64 画像として投入）
- Caption A
- Caption B

評価軸（4軸推奨）:

1. 正確性（映像内容との一致）
2. 情報量（重要な動作・対象を含むか）
3. 簡潔性（冗長でないか）
4. 可読性（自然な文か）

出力は構造化 JSON:

```json
{
	"video_id": "clerk_001",
	"winner": "B",
	"score_a": 0.72,
	"score_b": 0.86,
	"reason": "Bの方が主要動作を具体的に含み、誤りが少ない。"
}
```

### 4.4 集計指標

- Win rate: `ft` が勝った割合
- 平均スコア差: `mean(score_ft - score_base)`
- 失敗ケース分析: 負けサンプル上位 N 件の理由を確認

---

## 5. 実装タスク分解（そのまま着手用）

1. `scripts/01_build_teacher_data.py`
2. `scripts/02_make_swift_dataset.py`
3. `scripts/03_infer_captions.py`
4. `scripts/04_eval_pairwise.py`

各スクリプトの責務:

- `01`: metadata を読み、強いモデルで teacher データを JSONL 保存
- `02`: teacher_raw を sm-swift 学習形式へ変換して train/valid 分割
- `03`: base/ft それぞれで推論し prediction JSONL 出力
- `04`: GPT 評価を実行して winner とスコアを保存し、最終集計を出す

---

## 6. 品質管理チェックリスト

- 学習データに重複動画が多すぎない
- `thinking` が英語で、かつ観察可能な対象/動作のみを扱っている
- `thinking` が冗長すぎない（1-3文に制限）
- `caption` が英語1文で、店員意図と客意図を両方含む
- train/valid/test のリークなし
- 評価プロンプトを固定（比較の公平性確保）

---

## 7. 次の実装順（推奨）

1. `01_build_teacher_data.py` を先に作る
2. 10-30本の小規模データで `02` と `sft` を疎通
3. `03` と `04` で自動評価パイプラインを完成
4. 問題なければデータを増やして本学習

この README をベースに、次は `scripts/01_build_teacher_data.py` から順に実装できます。
