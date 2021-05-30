package mat

import "log"

type MatrixCmplxDense struct {
	rows, cols int
	major      MatMajor
	data       []complex128
}

func NewMatrixCmplxDense(r, c int) MatrixCmplxDense {
	if r < 0 || c < 0 {
		log.Fatalf("row and column sizes must be >= 0: got r = %v and c = %v\n", r, c)
	}
	return MatrixCmplxDense{major: Row, rows: r, cols: c, data: make([]complex128, r*c)}
}

func NewMatrixCmplxDenseData(maj MatMajor, n int, d []complex128) MatrixCmplxDense {
	if !maj.IsValid() {
		log.Fatalf("major not recognized: got %v\n", maj)
	} else if len(d)%n != 0 {
		log.Fatalf("dimension incompatible with data: got %v for data length %v\n", n, len(d))
	} else if maj == Row {
		return MatrixCmplxDense{rows: n, cols: len(d) / n, major: maj, data: d}
	} else {
		return MatrixCmplxDense{rows: len(d) / n, cols: n, major: maj, data: d}
	}
	return NewMatrixCmplxDense(0, 0)
}

func (m MatrixCmplxDense) Copy() MatrixCmplxDense {
	x := NewMatrixCmplxDense(m.rows, m.cols)
	x.major = m.major
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			x.Set(i, j, m.Get(i, j))
		}
	}
	return x
}

func (m MatrixCmplxDense) Equal(x MatrixCmplxDense) bool {
	if (m.rows != x.rows) || (m.cols != x.cols) {
		return false
	}
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			if m.Get(i, j) != x.Get(i, j) {
				return false
			}
		}
	}
	return true
}

func (m *MatrixCmplxDense) Major(maj MatMajor) {
	if !maj.IsValid() {
		log.Fatalf("Major not recognized: got %v\n", maj)
	} else if maj == m.major {
		return
	}

	newmat := NewMatrixCmplxDense(m.cols, m.rows)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			newmat.Set(j, i, m.Get(i, j))
		}
	}
	m.major = maj
	m.data = newmat.data
}

func (m *MatrixCmplxDense) Set(r, c int, a complex128) {
	if m.major == Row {
		m.data[c+m.cols*r] = a
	} else if m.major == Col {
		m.data[r+m.rows*c] = a
	}
}

func (m MatrixCmplxDense) Get(r, c int) complex128 {
	if m.major == Row {
		return m.data[c+m.cols*r]
	}
	return m.data[r+m.rows*c]
}

func (m MatrixCmplxDense) GetCol(c int) VectorCmplxDense {
	t := m.Copy()
	t.Major(Col)

	return NewVectorCmplxDenseData(t.data[c*m.rows : (c+1)*m.rows])
}

func (m MatrixCmplxDense) GetRow(r int) VectorCmplxDense {
	t := m.Copy()
	t.Major(Row)

	return NewVectorCmplxDenseData(t.data[r*m.cols : (r+1)*m.cols])
}

func (m MatrixCmplxDense) GetConj(r, c int) complex128 {
	return conjc128(m.Get(r, c))
}

func (m MatrixCmplxDense) GetRe(r, c int) float64 {
	return real(m.Get(r, c))
}

func (m MatrixCmplxDense) GetIm(r, c int) float64 {
	return imag(m.Get(r, c))
}

func (m *MatrixCmplxDense) Conj(r, c int) {
	m.Set(r, c, conjc128(m.Get(r, c)))
}

func (m *MatrixCmplxDense) Add(r, c int, a complex128) {
	m.Set(r, c, m.Get(r, c)+a)
}

func (m *MatrixCmplxDense) Div(r, c int, a complex128) {
	m.Set(r, c, m.Get(r, c)/a)
}

func (m *MatrixCmplxDense) Mul(r, c int, a complex128) {
	m.Set(r, c, m.Get(r, c)*a)
}

func (m *MatrixCmplxDense) Sub(r, c int, a complex128) {
	m.Set(r, c, m.Get(r, c)-a)
}

func (m *MatrixCmplxDense) Zero() {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.Set(i, j, 0)
		}
	}
}

func (m MatrixCmplxDense) Size() int {
	return m.rows * m.cols
}

type VectorCmplxDense struct {
	size int
	data []complex128
}

func NewVectorCmplxDense(n int) VectorCmplxDense {
	if n < 0 {
		log.Fatalf("Vector size must be > 0: got %v\n", n)
	}
	return VectorCmplxDense{size: n, data: make([]complex128, n)}
}

func NewVectorCmplxDenseData(d []complex128) VectorCmplxDense {
	return VectorCmplxDense{size: len(d), data: d}
}

func (v VectorCmplxDense) Copy() VectorCmplxDense {
	x := NewVectorCmplxDense(v.size)
	for i := 0; i < v.size; i++ {
		x.Set(i, v.Get(i))
	}
	return x
}

func (v VectorCmplxDense) CopyOff(n int) VectorCmplxDense {
	if n < 0 {
		log.Fatalf("n must be greater than or equal to 0: got %v\n", n)
	} else if n >= v.size {
		log.Fatalf("n must be less than current vector size: have %v, got %v\n", v.size, n)
	}

	x := NewVectorCmplxDense(v.size - n)
	for i := 0; i < x.size; i++ {
		x.Set(i, v.Get(i+n))
	}
	return x
}

func (v *VectorCmplxDense) AddElem() {
	v.size++
	v.data = append(v.data, 0)
}

func (v *VectorCmplxDense) Flip() {
	for i := 0; i < v.size/2; i++ {
		v.Swap((v.size-1)-i, i)
	}
}

func (v *VectorCmplxDense) Swap(x, y int) {
	temp := v.Get(x)
	v.Set(x, v.Get(y))
	v.Set(y, temp)
}

func (v VectorCmplxDense) ToMatrix(maj MatMajor) MatrixCmplxDense {
	var m MatrixCmplxDense

	if !maj.IsValid() {
		log.Fatalf("major not recognized: got %v\n", maj.String())
	}

	if maj == Row {
		m = NewMatrixCmplxDense(v.size, 1)
		for i := 0; i < v.size; i++ {
			m.Set(i, 0, v.Get(i))
		}
	} else {
		m = NewMatrixCmplxDense(1, v.size)
		for i := 0; i < v.size; i++ {
			m.Set(0, i, v.Get(i))
		}
	}

	return m
}

func (v VectorCmplxDense) Equal(x VectorCmplxDense) bool {
	if v.size != x.size {
		return false
	}
	for i := 0; i < v.size; i++ {
		if v.Get(i) != x.Get(i) {
			return false
		}
	}
	return true
}

func (v *VectorCmplxDense) Set(n int, a complex128) {
	v.data[n] = a
}

func (v VectorCmplxDense) Get(n int) complex128 {
	return v.data[n]
}

func (v VectorCmplxDense) GetConj(n int) complex128 {
	return conjc128(v.data[n])
}

func (v VectorCmplxDense) GetRe(n int) float64 {
	return real(v.data[n])
}

func (v VectorCmplxDense) GetIm(n int) float64 {
	return imag(v.data[n])
}

func (v *VectorCmplxDense) Conj(n int) {
	v.Set(n, conjc128(v.Get(n)))
}

func (v *VectorCmplxDense) Add(n int, a complex128) {
	v.Set(n, v.Get(n)+a)
}

func (v *VectorCmplxDense) Div(n int, a complex128) {
	v.Set(n, v.Get(n)/a)
}

func (v *VectorCmplxDense) Mul(n int, a complex128) {
	v.Set(n, v.Get(n)*a)
}

func (v *VectorCmplxDense) Sub(n int, a complex128) {
	v.Set(n, v.Get(n)-a)
}

func (v *VectorCmplxDense) Zero() {
	v.data = make([]complex128, v.size)
}
