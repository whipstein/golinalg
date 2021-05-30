package mat

import (
	"bytes"
	"encoding/gob"
)

type matrixDenseGeneral struct {
	rows, cols int
	data       []float64
	opts       *MatOpts
}

func (m *matrixDenseGeneral) Get(r, c int) float64 {
	rnew, cnew := r, c

	// if m.validate {
	// 	if r > m.rows || c > m.cols || r < 0 || c < 0 {
	// 		// log.Fatalf("MatrixDense.Get: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 		log.Panicf("MatrixDense.Get: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 	}
	// }

	return m.data[getIdx(m.opts.Major, m.rows, m.cols, rnew, cnew)]
}

func (m *matrixDenseGeneral) Set(r, c int, x float64) {
	// if m.validate {
	// 	if r > m.rows || c > m.cols || r < 0 || c < 0 {
	// 		// log.Fatalf("MatrixDense.Set: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 		log.Panicf("MatrixDense.Set: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 	}
	// }

	m.data[getIdx(m.opts.Major, m.rows, m.cols, r, c)] = x
}

func (m *matrixDenseGeneral) Size() (int, int) {
	return m.rows, m.cols
}

func (m *matrixDenseGeneral) Data() []float64 {
	return m.data
}

func (m *matrixDenseGeneral) T() []float64 {
	t := NewTransBuilder(m.rows, m.cols, m.data)
	return t.T()
}

func (m *matrixDenseGeneral) Major() MatMajor {
	return m.opts.Major
}

func (m *matrixDenseGeneral) Opts() *MatOpts {
	return m.opts
}

func (m *matrixDenseGeneral) Copy(idx int) *matrixDenseGeneral {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := matrixDenseGeneral{}
	_ = d.Decode(&result)
	result.data = m.data[idx:]
	return &result
}

func (m *matrixDenseGeneral) DeepCopy() *matrixDenseGeneral {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := matrixDenseGeneral{}
	_ = d.Decode(&result)
	return &result
}

type matrixDenseSymmetric struct {
	rows, cols int
	data       []float64
	opts       *MatOpts
}

func (m *matrixDenseSymmetric) Get(r, c int) float64 {
	rnew, cnew := r, c

	// if m.validate {
	// 	if r > m.rows || c > m.cols || r < 0 || c < 0 {
	// 		// log.Fatalf("MatrixDense.Get: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 		log.Panicf("MatrixDense.Get: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 	}
	// }

	return m.data[getIdx(m.opts.Major, m.rows, m.cols, rnew, cnew)]
}

func (m *matrixDenseSymmetric) Set(r, c int, x float64) {
	// if m.validate {
	// 	if r > m.rows || c > m.cols || r < 0 || c < 0 {
	// 		// log.Fatalf("MatrixDense.Set: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 		log.Panicf("MatrixDense.Set: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 	}
	// }

	m.data[getIdx(m.opts.Major, m.rows, m.cols, r, c)] = x
}

func (m *matrixDenseSymmetric) Size() (int, int) {
	return m.rows, m.cols
}

func (m *matrixDenseSymmetric) Data() []float64 {
	return m.data
}

func (m *matrixDenseSymmetric) T() []float64 {
	t := NewTransBuilder(m.rows, m.cols, m.data)
	return t.T()
}

func (m *matrixDenseSymmetric) Major() MatMajor {
	return m.opts.Major
}

func (m *matrixDenseSymmetric) Opts() *MatOpts {
	return m.opts
}

func (m *matrixDenseSymmetric) Copy(idx int) *matrixDenseSymmetric {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := matrixDenseSymmetric{}
	_ = d.Decode(&result)
	result.data = m.data[idx:]
	return &result
}

func (m *matrixDenseSymmetric) DeepCopy() *matrixDenseSymmetric {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := matrixDenseSymmetric{}
	_ = d.Decode(&result)
	return &result
}

type matrixDenseTriangular struct {
	rows, cols int
	data       []float64
	opts       *MatOpts
}

func (m *matrixDenseTriangular) Get(r, c int) float64 {
	rnew, cnew := r, c

	// if m.validate {
	// 	if r > m.rows || c > m.cols || r < 0 || c < 0 {
	// 		// log.Fatalf("MatrixDense.Get: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 		log.Panicf("MatrixDense.Get: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 	}
	// }

	return m.data[getIdx(m.opts.Major, m.rows, m.cols, rnew, cnew)]
}

func (m *matrixDenseTriangular) Set(r, c int, x float64) {
	// if m.validate {
	// 	if r > m.rows || c > m.cols || r < 0 || c < 0 {
	// 		// log.Fatalf("MatrixDense.Set: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 		log.Panicf("MatrixDense.Set: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 	}
	// }

	m.data[getIdx(m.opts.Major, m.rows, m.cols, r, c)] = x
}

func (m *matrixDenseTriangular) Size() (int, int) {
	return m.rows, m.cols
}

func (m *matrixDenseTriangular) Data() []float64 {
	return m.data
}

func (m *matrixDenseTriangular) T() []float64 {
	t := NewTransBuilder(m.rows, m.cols, m.data)
	return t.T()
}

func (m *matrixDenseTriangular) Major() MatMajor {
	return m.opts.Major
}

func (m *matrixDenseTriangular) Opts() *MatOpts {
	return m.opts
}

func (m *matrixDenseTriangular) Copy(idx int) *matrixDenseTriangular {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := matrixDenseTriangular{}
	_ = d.Decode(&result)
	result.data = m.data[idx:]
	return &result
}

func (m *matrixDenseTriangular) DeepCopy() *matrixDenseTriangular {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := matrixDenseTriangular{}
	_ = d.Decode(&result)
	return &result
}
