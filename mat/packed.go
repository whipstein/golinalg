package mat

import (
	"bytes"
	"encoding/gob"
)

type matrixPackedSymmetric struct {
	rows, cols int
	data       []float64
	opts       *MatOpts
}

func (m *matrixPackedSymmetric) Get(idx, c int) float64 {
	// if m.validate {
	// 	if r > m.rows || c > m.cols || r < 0 || c < 0 {
	// 		// log.Fatalf("MatrixDense.Get: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 		log.Panicf("MatrixDense.Get: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 	}
	// }

	return m.data[idx]
}

func (m *matrixPackedSymmetric) Set(r, c int, x float64) {
	// if m.validate {
	// 	if r > m.rows || c > m.cols || r < 0 || c < 0 {
	// 		// log.Fatalf("MatrixDense.Set: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 		log.Panicf("MatrixDense.Set: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 	}
	// }

	m.data[getIdx(m.opts.Major, m.rows, m.cols, r, c)] = x
}

func (m *matrixPackedSymmetric) Size() (int, int) {
	return m.rows, m.cols
}

func (m *matrixPackedSymmetric) Data() []float64 {
	return m.data
}

func (m *matrixPackedSymmetric) T() []float64 {
	t := NewTransBuilder(m.rows, m.cols, m.data)
	return t.T()
}

func (m *matrixPackedSymmetric) Major() MatMajor {
	return m.opts.Major
}

func (m *matrixPackedSymmetric) Opts() *MatOpts {
	return m.opts
}

func (m *matrixPackedSymmetric) Copy(idx int) *matrixPackedSymmetric {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := matrixPackedSymmetric{}
	_ = d.Decode(&result)
	result.data = m.data[idx:]
	return &result
}

func (m *matrixPackedSymmetric) DeepCopy() *matrixPackedSymmetric {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := matrixPackedSymmetric{}
	_ = d.Decode(&result)
	return &result
}

type matrixPackedTriangular struct {
	rows, cols int
	data       []float64
	opts       *MatOpts
}

func (m *matrixPackedTriangular) Get(idx, c int) float64 {
	// if m.validate {
	// 	if r > m.rows || c > m.cols || r < 0 || c < 0 {
	// 		// log.Fatalf("MatrixDense.Get: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 		log.Panicf("MatrixDense.Get: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 	}
	// }

	return m.data[idx]
}

func (m *matrixPackedTriangular) Set(idx, c int, x float64) {
	// if m.validate {
	// 	if r > m.rows || c > m.cols || r < 0 || c < 0 {
	// 		// log.Fatalf("MatrixDense.Set: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 		log.Panicf("MatrixDense.Set: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 	}
	// }

	m.data[idx] = x
}

func (m *matrixPackedTriangular) Size() (int, int) {
	return m.rows, m.cols
}

func (m *matrixPackedTriangular) Data() []float64 {
	return m.data
}

func (m *matrixPackedTriangular) T() []float64 {
	t := NewTransBuilder(m.rows, m.cols, m.data)
	return t.T()
}

func (m *matrixPackedTriangular) Major() MatMajor {
	return m.opts.Major
}

func (m *matrixPackedTriangular) Opts() *MatOpts {
	return m.opts
}

func (m *matrixPackedTriangular) Copy(idx int) *matrixPackedTriangular {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := matrixPackedTriangular{}
	_ = d.Decode(&result)
	result.data = m.data[idx:]
	return &result
}

func (m *matrixPackedTriangular) DeepCopy() *matrixPackedTriangular {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := matrixPackedTriangular{}
	_ = d.Decode(&result)
	return &result
}
