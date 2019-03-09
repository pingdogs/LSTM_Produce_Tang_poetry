# LSTM_Produce_Tang_Poetry

>  穿越時空的詩人 - 用PyTorch重現李白的神經網絡

本專案主要以LSTM和Seq2seq模型建立李白詩詞產生器，以過往李白 所發表之詩詞作為訓練資料，最終產生五言及七言 絕句、律詩

● seq2seq模型是以編碼(Encode)和解碼(Decode)為代表 的架構方式，seq2seq模型是根據輸入Sequence X來生成 輸出Sequence Y

● 以encode和decode為代表的seq2seq模型，encode意思 是將輸入Sequence轉化成一個固定長度的向量，decode 意思是將輸入的固定長度向量解碼成輸出Sequence。

最後產生詩詞時，押韻則使用北曲新譜的韻部來做韻腳的選擇(分類詳見rhyme.json) 

產生結果：

一去不可識，高風生綠水，天地何所用，妾家本無催。(五絕）

漢宮天子開帝王，明月不見吳江水，天下白雪照雲雨，青天下有餘鳥啼，此地一朝別離居，天地一回玉階日，美人不肯相思君，君不見月上時回。（七律)
