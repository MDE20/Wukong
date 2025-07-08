package filter

import (
	"../wukong"
)

const Threads int = 12

type FilterFunc func(freq int, trainTracks [][]wukong.Detection, labels []bool, cfg map[string]string) Filter

type Filter interface {
	Predict(valTracks [][]wukong.Detection) []float64
	Close()
}

var FilterMap = make(map[string]FilterFunc)
