// ・frugally-deepはpythonで書いたtensorflow/kerasの学習済みモデルを読み込んでC++上で実行するライブラリ
// https://zenn.dev/mattn/articles/e871dab58be002
// ・h5ファイルをjsonファイルに変換することで読み込んで実行することができる、keras_export/convert_model_pyを使用
// ・


#include <iostream>
#include <fdeep/fdeep.hpp>

using namespace std;

int main(){
    // jsonファイルに変換した学習済みモデル読み込み
    const auto model=fdeep::load_model("mask_model.json");
    return 0;
}