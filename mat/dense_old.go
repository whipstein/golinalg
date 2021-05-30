package mat

// _type MatrixDense struct {
// 	rows, cols int
// 	major      Major
// 	data       []float64
// }

// func NewMatrixDense(r, c int) *MatrixDense {
// 	if r < 0 || c < 0 {
// 		log.Fatalf("row and column sizes must be >= 0: got r = %v and c = %v\n", r, c)
// 	}
// 	return &MatrixDense{major: Row, rows: r, cols: c, data: make([]float64, r*c)}
// }

// func NewMatrixDenseData(maj Major, n int, d []float64) *MatrixDense {
// 	if !maj.IsValid() {
// 		log.Fatalf("major not recognized: got %v\n", maj)
// 	} else if len(d)%n != 0 {
// 		log.Fatalf("dimension incompatible with data: got %v for data length %v\n", n, len(d))
// 	} else if maj == Row {
// 		return &MatrixDense{rows: n, cols: len(d) / n, major: maj, data: d}
// 	} else {
// 		return &MatrixDense{rows: len(d) / n, cols: n, major: maj, data: d}
// 	}
// 	return NewMatrixDense(0, 0)
// }

// func (m *MatrixDense) Copy() *MatrixDense {
// 	x := NewMatrixDense(m.rows, m.cols)
// 	x.major = m.major
// 	for i := 0; i < m.rows; i++ {
// 		for j := 0; j < m.cols; j++ {
// 			x.Set(i, j, m.Get(i, j))
// 		}
// 	}
// 	return x
// }

// func (m *MatrixDense) Equal(x *MatrixDense) bool {
// 	if (m.rows != x.rows) || (m.cols != x.cols) {
// 		return false
// 	}
// 	for i := 0; i < m.rows; i++ {
// 		for j := 0; j < m.cols; j++ {
// 			if m.Get(i, j) != x.Get(i, j) {
// 				return false
// 			}
// 		}
// 	}
// 	return true
// }

// func (m *MatrixDense) ToMajor(maj Major) {
// 	if !maj.IsValid() {
// 		log.Fatalf("Major not recognized: got %v\n", maj)
// 	} else if maj == m.major {
// 		return
// 	}

// 	newmat := NewMatrixDense(m.cols, m.rows)
// 	for i := 0; i < m.rows; i++ {
// 		for j := 0; j < m.cols; j++ {
// 			newmat.Set(j, i, m.Get(i, j))
// 		}
// 	}
// 	m.major = maj
// 	m.data = newmat.data
// }

// func (m *MatrixDense) Set(r, c int, x float64) {
// 	if m.major == Row {
// 		m.data[c+m.cols*r] = x
// 	} else if m.major == Col {
// 		m.data[r+m.rows*c] = x
// 	}
// }

// func (m *MatrixDense) Get(r, c int) float64 {
// 	if m.major == Row {
// 		return m.data[c+m.cols*r]
// 	} else if m.major == Col {
// 		return m.data[r+m.rows*c]
// 	}
// 	return 0
// }

// func (m *MatrixDense) Add(r, c int, a float64) {
// 	m.Set(r, c, m.Get(r, c)+a)
// }

// func (m *MatrixDense) Div(r, c int, a float64) {
// 	m.Set(r, c, m.Get(r, c)/a)
// }

// func (m *MatrixDense) Mul(r, c int, a float64) {
// 	m.Set(r, c, m.Get(r, c)*a)
// }

// func (m *MatrixDense) Sub(r, c int, a float64) {
// 	m.Set(r, c, m.Get(r, c)-a)
// }

// func (m *MatrixDense) Zero() {
// 	for i := 0; i < m.rows; i++ {
// 		for j := 0; j < m.cols; j++ {
// 			m.Set(i, j, 0)
// 		}
// 	}
// }

// func (m *MatrixDense) Size() int {
// 	return m.rows * m.cols
// }

// func (m *MatrixDense) Rows() int {
// 	return m.rows
// }

// func (m *MatrixDense) Cols() int {
// 	return m.cols
// }

// func (m *MatrixDense) Data() []float64 {
// 	return m.data
// }

// _type VectorDense struct {
// 	size   int
// 	stride int
// 	data   []float64
// }

// func NewVectorDense(n int) *VectorDense {
// 	if n < 0 {
// 		log.Fatalf("Vector size must be > 0: got %v\n", n)
// 	}
// 	return &VectorDense{size: n, stride: 1, data: make([]float64, n)}
// }

// func NewVectorDenseData(d []float64) *VectorDense {
// 	return &VectorDense{size: len(d), stride: 1, data: d}
// }

// func (v *VectorDense) Copy() *VectorDense {
// 	x := NewVectorDense(v.size)
// 	for i := 0; i < v.size; i++ {
// 		x.Set(i, v.Get(i))
// 	}
// 	return x
// }

// func (v *VectorDense) CopyOff(n int) *VectorDense {
// 	if n < 0 {
// 		log.Fatalf("n must be greater than or equal to 0: got %v\n", n)
// 	} else if n >= v.size {
// 		log.Fatalf("n must be less than current vector size: have %v, got %v\n", v.size, n)
// 	}

// 	x := NewVectorDense(v.size - n)
// 	for i := 0; i < x.size; i++ {
// 		x.Set(i, v.Get(i+n))
// 	}
// 	return x
// }

// func (v *VectorDense) AddElem() {
// 	v.size++
// 	v.data = append(v.data, 0)
// }

// func (v *VectorDense) Flip() {
// 	for i := 0; i < v.size/2; i++ {
// 		v.Swap((v.size-1)-i, i)
// 	}
// }

// func (v *VectorDense) Swap(x, y int) {
// 	temp := v.Get(x)
// 	v.Set(x, v.Get(y))
// 	v.Set(y, temp)
// }

// func (v *VectorDense) ToMatrix(maj Major) *MatrixDense {
// 	var m *MatrixDense

// 	if !maj.IsValid() {
// 		log.Fatalf("major not recognized: got %v\n", maj.String())
// 	}

// 	if maj == Row {
// 		m = NewMatrixDense(v.size, 1)
// 		for i := 0; i < v.size; i++ {
// 			m.Set(i, 0, v.Get(i))
// 		}
// 	} else {
// 		m = NewMatrixDense(1, v.size)
// 		for i := 0; i < v.size; i++ {
// 			m.Set(0, i, v.Get(i))
// 		}
// 	}

// 	return m
// }

// func (v *VectorDense) Equal(x *VectorDense) bool {
// 	if v.size != x.size {
// 		return false
// 	}
// 	for i := 0; i < v.size; i++ {
// 		if v.Get(i) != x.Get(i) {
// 			return false
// 		}
// 	}
// 	return true
// }

// func (v *VectorDense) Set(n int, a float64) {
// 	v.data.Set(n,a)
// }

// func (v *VectorDense) Get(n int) float64 {
// 	return v.data[n]
// }

// func (v *VectorDense) Add(n int, a float64) {
// 	v.Set(n, v.Get(n)+a)
// }

// func (v *VectorDense) Div(n int, a float64) {
// 	v.Set(n, v.Get(n)/a)
// }

// func (v *VectorDense) Mul(n int, a float64) {
// 	v.Set(n, v.Get(n)*a)
// }

// func (v *VectorDense) Sub(n int, a float64) {
// 	v.Set(n, v.Get(n)-a)
// }

// func (v *VectorDense) Size() int {
// 	return v.size
// }

// func (v *VectorDense) Stride() int {
// 	return v.stride
// }

// func (v *VectorDense) Data() []float64 {
// 	return v.data
// }

// func (v *VectorDense) Zero() {
// 	v.data = make([]float64, v.size)
// }
