package mat

import (
	"math/rand"
	"testing"
)

var mf = MatrixFactory()
var cmf = CMatrixFactory()

func TestNewMatOpts(t *testing.T) {
	x := NewMatOpts()

	if x.Style != General {
		t.Errorf("MatOpts: Style: got %s want General\n", x.Style)
	}
	if x.Storage != Dense {
		t.Errorf("MatOpts: Storage: got %s want Dense\n", x.Storage)
	}
	if x.Uplo != Full {
		t.Errorf("MatOpts: Uplo: got %s want Full\n", x.Uplo)
	}
	if x.Diag != NonUnit {
		t.Errorf("MatOpts: Diag: got %s want NonUnit\n", x.Diag)
	}
	if x.Side != Left {
		t.Errorf("MatOpts: Side: got %s want Left\n", x.Side)
	}
	if x.Major != Row {
		t.Errorf("MatOpts: Major: got %s want Row\n", x.Major)
	}

	y := NewMatOptsCol()

	if y.Style != General {
		t.Errorf("MatOpts: Style: got %s want General\n", y.Style)
	}
	if y.Storage != Dense {
		t.Errorf("MatOpts: Storage: got %s want Dense\n", y.Storage)
	}
	if y.Uplo != Full {
		t.Errorf("MatOpts: Uplo: got %s want Full\n", y.Uplo)
	}
	if y.Diag != NonUnit {
		t.Errorf("MatOpts: Diag: got %s want NonUnit\n", y.Diag)
	}
	if y.Side != Left {
		t.Errorf("MatOpts: Side: got %s want Left\n", y.Side)
	}
	if y.Major != Col {
		t.Errorf("MatOpts: Major: got %s want Col\n", y.Major)
	}

	z := x.DeepCopy()
	z.Major = Col
	if x.Major != Row {
		t.Errorf("MatOpts: Major: got %s want Row\n", x.Major)
	}
	if z.Major != Col {
		t.Errorf("MatOpts: Major: got %s want Col\n", z.Major)
	}
}

func TestMatOptions(t *testing.T) {
	// Test MatStyle
	{
		var x MatStyle

		for _, val := range IterMatStyle() {
			x = val
			if x.String() != val.String() || !x.IsValid() {
				t.Errorf("MatStyle: got %s, want %s\n", x, val)
			}
		}
	}

	// Test MatStorage
	{
		var x MatStorage

		for _, val := range IterMatStorage() {
			x = val
			if x.String() != val.String() || !x.IsValid() {
				t.Errorf("MatStorage: got %s, want %s\n", x, val)
			}
		}
	}

	// Test MatMajor
	{
		var x MatMajor

		for _, val := range IterMatMajor() {
			x = val
			if x.String() != val.String() || !x.IsValid() {
				t.Errorf("MatMajor: got %s, want %s\n", x, val)
			}
		}
	}

	// Test MatTrans
	{
		var x MatTrans

		for _, val := range IterMatTrans() {
			x = val
			if x.String() != val.String() || !x.IsValid() {
				t.Errorf("MatTrans: got %s, want %s\n", x, val)
			}
			if x.String() != "NoTrans" && !x.IsTrans() {
				t.Errorf("MatTrans: IsTrans: got %s, want %s\n", x, val)
			}
		}
		for _, val := range []byte{'n', 'N', 't', 'T', 'c', 'C'} {
			x = TransByte(val)
			if !x.IsValid() {
				t.Errorf("MatTrans: TransByte: got %s, want %c\n", x, val)
			}
		}
	}

	// Test MatUplo
	{
		var x MatUplo

		for _, val := range IterMatUplo() {
			x = val
			if x.String() != val.String() || !x.IsValid() {
				t.Errorf("MatUplo: got %s, want %s\n", x, val)
			}
		}
		for _, val := range []byte{'u', 'U', 'l', 'L', 'f', 'F', ' '} {
			x = UploByte(val)
			if !x.IsValid() {
				t.Errorf("MatUplo: UploByte: got %s, want %c\n", x, val)
			}
		}
	}

	// Test MatDiag
	{
		var x MatDiag

		for _, val := range IterMatDiag() {
			x = val
			if x.String() != val.String() || !x.IsValid() {
				t.Errorf("MatDiag: got %s, want %s\n", x, val)
			}
		}
		for _, val := range []byte{'n', 'N', 'u', 'U'} {
			x = DiagByte(val)
			if !x.IsValid() {
				t.Errorf("MatDiag: DiagByte: got %s, want %c\n", x, val)
			}
		}
	}

	// Test MatSide
	{
		var x MatSide

		for _, val := range IterMatSide() {
			x = val
			if x.String() != val.String() || !x.IsValid() {
				t.Errorf("MatSide: got %s, want %s\n", x, val)
			}
		}
		for _, val := range []byte{'l', 'L', 'r', 'R'} {
			x = SideByte(val)
			if !x.IsValid() {
				t.Errorf("MatSide: SideByte: got %s, want %c\n", x, val)
			}
		}
	}
}

func TestMatrixAppendCol(t *testing.T) {
	m := mf(4, 5)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Set(i, j, float64(j+m.Cols*i))
		}
	}
	d := make([]float64, 4)
	for i := 0; i < len(d); i++ {
		d[i] = float64(i + m.Rows*m.Cols)
	}
	m.AppendCol(d)
	for i := 0; i < len(d); i++ {
		if d[i] != m.Get(i, m.Cols-1) {
			t.Errorf("AppendCol: got %v want %v", m.Get(i, m.Cols-1), d[i])
		}
	}
	m = mf(4, 5)
	m.ToColMajor()
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Set(i, j, float64(j+m.Cols*i))
		}
	}
	m.AppendCol(d)
	for i := 0; i < len(d); i++ {
		if d[i] != m.Get(i, m.Cols-1) {
			t.Errorf("AppendCol: got %v want %v", m.Get(i, m.Cols-1), d[i])
		}
	}
}
func BenchmarkMatrixAppendCol(b *testing.B) {
	d := make([]float64, 10000)
	for i := 0; i < len(d); i++ {
		d[i] = rand.Float64()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m := mf(10000, 10000)
		m.AppendCol(d)
	}
}

func TestMatrixAppendRow(t *testing.T) {
	m := mf(4, 5)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Set(i, j, float64(j+m.Cols*i))
		}
	}
	d := make([]float64, 5)
	for i := 0; i < len(d); i++ {
		d[i] = float64(i + m.Rows*m.Cols)
	}
	m.AppendRow(d)
	for i := 0; i < len(d); i++ {
		if d[i] != m.Get(m.Rows-1, i) {
			t.Errorf("AppendRow: got %v want %v", m.Get(m.Rows-1, i), d[i])
		}
	}
	m = mf(4, 5)
	m.ToColMajor()
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Set(i, j, float64(j+m.Cols*i))
		}
	}
	m.AppendRow(d)
	for i := 0; i < len(d); i++ {
		if d[i] != m.Get(m.Rows-1, i) {
			t.Errorf("AppendRow: got %v want %v", m.Get(m.Rows-1, i), d[i])
		}
	}
}
func BenchmarkMatrixAppendRow(b *testing.B) {
	d := make([]float64, 10000)
	for i := 0; i < len(d); i++ {
		d[i] = rand.Float64()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m := mf(10000, 10000)
		m.AppendRow(d)
	}
}

func BenchmarkMatrixGet(b *testing.B) {
	m := mf(10000, 10000)
	x := rand.Intn(10000)
	y := rand.Intn(10000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		z := m.Get(x, y)
		_ = z
	}
}

func BenchmarkCMatrixGet(b *testing.B) {
	m := cmf(10000, 10000)
	x := rand.Intn(10000)
	y := rand.Intn(10000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		z := m.Get(x, y)
		_ = z
	}
}

func BenchmarkMatrixSet(b *testing.B) {
	m := mf(10000, 10000)
	x := rand.Intn(10000)
	y := rand.Intn(10000)
	z := rand.Float64()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Set(x, y, z)
	}
}

func TestMatrixSwap(t *testing.T) {
	m1 := mf(10000, 10000)
	for i := 0; i < m1.Rows; i++ {
		for j := 0; j < m1.Cols; j++ {
			m1.Set(i, j, float64(j+m1.Cols*i))
		}
	}
	m2 := m1.DeepCopy()
	for i := 0; i < m1.Rows/2; i++ {
		for j := 0; j < m1.Cols; j++ {
			m1.Swap(i, j, m1.Rows-1-i, m1.Cols-1-j)
		}
	}
	for i := 0; i < m1.Rows; i++ {
		for j := 0; j < m1.Cols; j++ {
			if m1.Get(i, j) != m2.Get(m2.Rows-1-i, m2.Cols-1-j) {
				t.Errorf("Swap: got %v want %v", m1.Get(i, j), m2.Get(m2.Rows-1-i, m2.Cols-1-j))
			}
		}
	}
}
func BenchmarkMatrixSwap(b *testing.B) {
	m := mf(10000, 10000)
	x1 := rand.Intn(10000)
	y1 := rand.Intn(10000)
	x2 := rand.Intn(10000)
	y2 := rand.Intn(10000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Swap(x1, y1, x2, y2)
	}
}

func BenchmarkCMatrixSet(b *testing.B) {
	m := cmf(10000, 10000)
	x := rand.Intn(10000)
	y := rand.Intn(10000)
	zr := rand.Float64()
	zi := rand.Float64()
	z := complex(zr, zi)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Set(x, y, z)
	}
}
