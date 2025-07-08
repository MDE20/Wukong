package filter

import (
	"fmt"

	"github.com/zhengpeijun/miris-master/miris"
	rnnlib "github.com/zhengpeijun/miris-master/models/rnn"
)

func init() {
	FilterMap["rnn"] = MakeRNNFilter
}

type RNNFilter struct {
	model rnnlib.Model
}

func MakeRNNFilter(freq int, tracks [][]miris.Detection, labels []bool, cfg map[string]string) Filter {
	fmt.Println("[MakeRNNFilter] cfg:", cfg)
	fmt.Println("[MakeRNNFilter] cfg:", cfg["model_path"])
	model := rnnlib.MakeModel(1, cfg["model_path"])
	return RNNFilter{model}
}

func (f RNNFilter) Predict(tracks [][]miris.Detection) []float64 {
	outputs := f.model.Infer(tracks)
	scores := make([]float64, len(outputs))
	for i := range scores {
		scores[i] = outputs[i][0]
	}
	return scores
}

func (f RNNFilter) Close() {
	f.model.Close()
}
