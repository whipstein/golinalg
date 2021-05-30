package mat

import (
	"bytes"
	"encoding/gob"
)

type matrixBandedGeneral struct {
	rows, cols int
	data       []float64
	opts       *MatOpts
}

func (m *matrixBandedGeneral) Get(r, c int) float64 {
	rnew, cnew := r, c

	// if m.validate {
	// 	if r > m.rows || c > m.cols || r < 0 || c < 0 {
	// 		// log.Fatalf("MatrixDense.Get: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 		log.Panicf("MatrixDense.Get: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 	}
	// }

	return m.data[getIdx(m.opts.Major, m.rows, m.cols, rnew, cnew)]
}

func (m *matrixBandedGeneral) Set(r, c int, x float64) {
	// if m.validate {
	// 	if r > m.rows || c > m.cols || r < 0 || c < 0 {
	// 		// log.Fatalf("MatrixDense.Set: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 		log.Panicf("MatrixDense.Set: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 	}
	// }

	m.data[getIdx(m.opts.Major, m.rows, m.cols, r, c)] = x
}

func (m *matrixBandedGeneral) Size() (int, int) {
	return m.rows, m.cols
}

func (m *matrixBandedGeneral) Data() []float64 {
	return m.data
}

func (m *matrixBandedGeneral) T() []float64 {
	t := NewTransBuilder(m.rows, m.cols, m.data)
	return t.T()
}

func (m *matrixBandedGeneral) Major() MatMajor {
	return m.opts.Major
}

func (m *matrixBandedGeneral) Opts() *MatOpts {
	return m.opts
}

func (m *matrixBandedGeneral) K() (int, int) {
	return m.opts.Kl, m.opts.Ku
}

func (m *matrixBandedGeneral) Copy(idx int) *matrixBandedGeneral {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := matrixBandedGeneral{}
	_ = d.Decode(&result)
	result.data = m.data[idx:]
	return &result
}

func (m *matrixBandedGeneral) DeepCopy() *matrixBandedGeneral {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := matrixBandedGeneral{}
	_ = d.Decode(&result)
	return &result
}

type matrixBandedSymmetric struct {
	rows, cols int
	data       []float64
	opts       *MatOpts
}

func (m *matrixBandedSymmetric) Get(r, c int) float64 {
	rnew, cnew := r, c

	// if m.validate {
	// 	if r > m.rows || c > m.cols || r < 0 || c < 0 {
	// 		// log.Fatalf("MatrixDense.Get: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 		log.Panicf("MatrixDense.Get: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 	}
	// }

	return m.data[getIdx(m.opts.Major, m.rows, m.cols, rnew, cnew)]
}

func (m *matrixBandedSymmetric) Set(r, c int, x float64) {
	// if m.validate {
	// 	if r > m.rows || c > m.cols || r < 0 || c < 0 {
	// 		// log.Fatalf("MatrixDense.Set: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 		log.Panicf("MatrixDense.Set: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 	}
	// }

	m.data[getIdx(m.opts.Major, m.rows, m.cols, r, c)] = x
}

func (m *matrixBandedSymmetric) Size() (int, int) {
	return m.rows, m.cols
}

func (m *matrixBandedSymmetric) Data() []float64 {
	return m.data
}

func (m *matrixBandedSymmetric) T() []float64 {
	t := NewTransBuilder(m.rows, m.cols, m.data)
	return t.T()
}

func (m *matrixBandedSymmetric) Major() MatMajor {
	return m.opts.Major
}

func (m *matrixBandedSymmetric) Opts() *MatOpts {
	return m.opts
}

func (m *matrixBandedSymmetric) Copy(idx int) *matrixBandedSymmetric {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := matrixBandedSymmetric{}
	_ = d.Decode(&result)
	result.data = m.data[idx:]
	return &result
}

func (m *matrixBandedSymmetric) DeepCopy() *matrixBandedSymmetric {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := matrixBandedSymmetric{}
	_ = d.Decode(&result)
	return &result
}

type matrixBandedTriangular struct {
	rows, cols int
	data       []float64
	opts       *MatOpts
}

func (m *matrixBandedTriangular) Get(r, c int) float64 {
	rnew, cnew := r, c

	// if m.validate {
	// 	if r > m.rows || c > m.cols || r < 0 || c < 0 {
	// 		// log.Fatalf("MatrixDense.Get: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 		log.Panicf("MatrixDense.Get: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 	}
	// }

	return m.data[getIdx(m.opts.Major, m.rows, m.cols, rnew, cnew)]
}

func (m *matrixBandedTriangular) Set(r, c int, x float64) {
	// if m.validate {
	// 	if r > m.rows || c > m.cols || r < 0 || c < 0 {
	// 		// log.Fatalf("MatrixDense.Set: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 		log.Panicf("MatrixDense.Set: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
	// 	}
	// }

	m.data[getIdx(m.opts.Major, m.rows, m.cols, r, c)] = x
}

func (m *matrixBandedTriangular) Size() (int, int) {
	return m.rows, m.cols
}

func (m *matrixBandedTriangular) Data() []float64 {
	return m.data
}

func (m *matrixBandedTriangular) T() []float64 {
	t := NewTransBuilder(m.rows, m.cols, m.data)
	return t.T()
}

func (m *matrixBandedTriangular) Major() MatMajor {
	return m.opts.Major
}

func (m *matrixBandedTriangular) Opts() *MatOpts {
	return m.opts
}

func (m *matrixBandedTriangular) Copy(idx int) *matrixBandedTriangular {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := matrixBandedTriangular{}
	_ = d.Decode(&result)
	result.data = m.data[idx:]
	return &result
}

func (m *matrixBandedTriangular) DeepCopy() *matrixBandedTriangular {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := matrixBandedTriangular{}
	_ = d.Decode(&result)
	return &result
}
