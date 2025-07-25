package gnn

import (
	"../wukong"

	"bufio"
	"encoding/json"
	"io"
	"log"
	"os/exec"
	"strconv"
)

type GNN struct {
	cmd        *exec.Cmd
	stdin      io.WriteCloser
	rd         *bufio.Reader
	detections [][]wukong.Detection
}

func NewGNN(modelPath string, detectionPath string, framePath string, frameScale int) *GNN {
	var detections [][]wukong.Detection
	wukong.ReadJSON(detectionPath, &detections)

	cmd := exec.Command("python", "models/gnn/wrapper.py", modelPath, detectionPath, framePath, strconv.Itoa(frameScale))
	stdin, err := cmd.StdinPipe()
	if err != nil {
		panic(err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		panic(err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		panic(err)
	}
	err = cmd.Start()
	if err != nil {
		panic(err)
	}

	go wukong.LogStderr("gnn", stderr)
	rd := bufio.NewReader(stdout)
	return &GNN{cmd, stdin, rd, detections}
}

func (gnn *GNN) NumFrames() int {
	return len(gnn.detections)
}

func (gnn *GNN) Infer(idx1 int, idx2 int) [][]float64 {
	return gnn.InferMany([][2]int{{idx1, idx2}}, "")[0]
}

// frames [][2]int：输入的帧索引数组，每个元素是一个包含两个整数的元组 [start, end]，表示帧范围。
// logPrefix string：日志前缀，用于调试和跟踪处理进度。
func (gnn *GNN) InferMany(frames [][2]int, logPrefix string) [][][]float64 {
	// 用于存储所有帧推理的结果，最终返回给调用者
	var mats [][][]float64
	for i := 0; i < len(frames); i += 16 {
		if logPrefix != "" && i%16 == 0 {
			log.Printf(" %d/%d", frames[i][0], frames[i][1])
			log.Printf(logPrefix+" %d/%d (%d/%d)", frames[i][0], len(gnn.detections), i, len(frames))
		}
		end := i + 16
		if end > len(frames) {
			end = len(frames)
		}
		curFrames := frames[i:end]

		// 增加调试日志，打印发送到子进程的帧信息
		//log.Printf("Sending frames to subprocess: %v", curFrames)

		bytes, err := json.Marshal(curFrames)
		if err != nil {
			panic(err)
		}
		if _, err := gnn.stdin.Write([]byte(string(bytes) + "\n")); err != nil {
			panic(err)
		}
		line, err := gnn.rd.ReadString('\n')
		if err != nil {
			panic(err)
		}
		var curMats [][][]float64
		if err := json.Unmarshal([]byte(line), &curMats); err != nil {
			panic(err)
		}
		mats = append(mats, curMats...)
	}
	return mats
}

func (gnn *GNN) Close() {
	gnn.stdin.Close()
	gnn.cmd.Wait()
}
