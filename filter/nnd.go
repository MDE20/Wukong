package filter

import (
	"../wukong"

	"log"
)

func init() {
	FilterMap["nnd"] = MakeNNDFilter
}

type NNDFilter struct {
	tracks [][]wukong.Detection
}

func MakeNNDFilter(freq int, tracks [][]wukong.Detection, labels []bool, cfg map[string]string) Filter {
	var trueTracks [][]wukong.Detection
	for i, track := range tracks {
		if labels[i] && len(track) >= 2 {
			trueTracks = append(trueTracks, track)
		}
	}
	return NNDFilter{trueTracks}
}

func (nnd NNDFilter) Predict(tracks [][]wukong.Detection) []float64 {
	ch := make(chan int)
	donech := make(chan map[int]float64)
	for i := 0; i < Threads; i++ {
		go func() {
			m := make(map[int]float64)
			for idx := range ch {
				track := tracks[idx]
				if len(track) <= 1 {
					m[idx] = -1000
					continue
				}
				var bestDistance float64 = -1
				for _, track2 := range nnd.tracks {
					d := wukong.TrackDistance(track, track2)
					if bestDistance == -1 || d < bestDistance {
						bestDistance = d
					}
				}
				m[idx] = -bestDistance
			}
			donech <- m
		}()
	}
	for i := range tracks {
		if i%1000 == 0 {
			log.Printf("[filter-nnd] ... %d/%d", i, len(tracks))
		}
		ch <- i
	}
	close(ch)
	scores := make([]float64, len(tracks))
	for i := 0; i < Threads; i++ {
		m := <-donech
		for idx, score := range m {
			scores[idx] = score
		}
	}
	return scores
}

func (nnd NNDFilter) Close() {}
