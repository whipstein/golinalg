package mat

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"log"
	"math"
	"math/cmplx"
)

func getIdx(m MatMajor, rows, cols, r, c int) int {
	if m == Row {
		return c + cols*r
	}
	return r + rows*c
}

type MatStyle int

const (
	General MatStyle = iota
	Symmetric
	Hermitian
	Triangular
)

func (m MatStyle) String() string {
	switch m {
	case General:
		return "General"
	case Symmetric:
		return "Symmetric"
	case Hermitian:
		return "Hermitian"
	case Triangular:
		return "Triangular"
	}
	return fmt.Sprintf("Unrecognized: %c", byte(m))
}
func (m MatStyle) IsValid() bool {
	if m == General || m == Symmetric || m == Hermitian || m == Triangular {
		return true
	}
	return false
}
func IterMatStyle() []MatStyle {
	return []MatStyle{General, Symmetric, Hermitian, Triangular}
}

type MatStorage int

const (
	Dense MatStorage = iota
	Banded
	Packed
)

func (m MatStorage) String() string {
	switch m {
	case Dense:
		return "Dense"
	case Banded:
		return "Banded"
	case Packed:
		return "Packed"
	}
	return fmt.Sprintf("Unrecognized: %c", byte(m))
}
func (m MatStorage) IsValid() bool {
	if m == Dense || m == Banded || m == Packed {
		return true
	}
	return false
}
func IterMatStorage() []MatStorage {
	return []MatStorage{Dense, Banded, Packed}
}

type MatMajor int

const (
	Row MatMajor = iota
	Col
)

func (m MatMajor) String() string {
	switch m {
	case Row:
		return "Row"
	case Col:
		return "Col"
	}
	return fmt.Sprintf("Unrecognized: %c", byte(m))
}
func (m MatMajor) IsValid() bool {
	if m == Row || m == Col {
		return true
	}
	return false
}
func IterMatMajor() []MatMajor {
	return []MatMajor{Row, Col}
}

type MatTrans int

const (
	NoTrans MatTrans = iota
	Trans
	ConjTrans
)

func (t MatTrans) Byte() byte {
	switch t {
	case NoTrans:
		return byte('N')
	case Trans:
		return byte('T')
	case ConjTrans:
		return byte('C')
	}
	return byte(' ')
}
func (t MatTrans) String() string {
	switch t {
	case NoTrans:
		return "NoTrans"
	case Trans:
		return "Trans"
	case ConjTrans:
		return "ConjTrans"
	}
	return fmt.Sprintf("Unrecognized: %c", byte(t))
}
func (t MatTrans) IsValid() bool {
	if t == NoTrans || t == Trans || t == ConjTrans {
		return true
	}
	return false
}
func (t MatTrans) IsTrans() bool {
	if t == Trans || t == ConjTrans {
		return true
	}
	return false
}
func TransByte(b byte) MatTrans {
	switch b {
	case 'n':
		return NoTrans
	case 'N':
		return NoTrans
	case 't':
		return Trans
	case 'T':
		return Trans
	case 'c':
		return ConjTrans
	case 'C':
		return ConjTrans
	default:
		return -1
	}
}
func IterMatTrans(conj ...bool) []MatTrans {
	if conj == nil || conj[0] == true {
		return []MatTrans{NoTrans, Trans, ConjTrans}
	}

	return []MatTrans{NoTrans, Trans}
}

type MatUplo int

const (
	Full MatUplo = iota
	Lower
	Upper
)

func (u MatUplo) Byte() byte {
	switch u {
	case Full:
		return byte('F')
	case Lower:
		return byte('L')
	case Upper:
		return byte('U')
	}
	return byte(' ')
}
func (u MatUplo) String() string {
	switch u {
	case Lower:
		return "Lower"
	case Upper:
		return "Upper"
	}
	return fmt.Sprintf("Unrecognized: %c", byte(u))
}
func (u MatUplo) IsValid() bool {
	if u == Lower || u == Upper || u == Full {
		return true
	}
	return false
}
func UploByte(b byte) MatUplo {
	switch b {
	case 'u':
		return Upper
	case 'U':
		return Upper
	case 'l':
		return Lower
	case 'L':
		return Lower
	default:
		return Full
	}
}
func IterMatUplo(full ...bool) []MatUplo {
	if full == nil || full[0] == true {
		return []MatUplo{Upper, Lower, Full}
	}

	return []MatUplo{Upper, Lower}
}

type MatDiag int

const (
	NonUnit MatDiag = iota
	Unit
)

func (d MatDiag) Byte() byte {
	switch d {
	case NonUnit:
		return byte('N')
	case Unit:
		return byte('U')
	}
	return byte(' ')
}
func (d MatDiag) String() string {
	switch d {
	case NonUnit:
		return "NonUnit"
	case Unit:
		return "Unit"
	}
	return fmt.Sprintf("Unrecognized: %c", byte(d))
}
func (d MatDiag) IsValid() bool {
	if d == NonUnit || d == Unit {
		return true
	}
	return false
}
func DiagByte(b byte) MatDiag {
	switch b {
	case 'n':
		return NonUnit
	case 'N':
		return NonUnit
	case 'u':
		return Unit
	case 'U':
		return Unit
	default:
		return -1
	}
}
func IterMatDiag() []MatDiag {
	return []MatDiag{NonUnit, Unit}
}

type MatSide int

const (
	Left MatSide = iota
	Right
	Both
)

func (s MatSide) Byte() byte {
	switch s {
	case Left:
		return byte('L')
	case Right:
		return byte('R')
	case Both:
		return byte('B')
	}
	return byte(' ')
}
func (s MatSide) String() string {
	switch s {
	case Left:
		return "Left"
	case Right:
		return "Right"
	case Both:
		return "Both"
	}
	return fmt.Sprintf("Unrecognized: %c", byte(s))
}
func (s MatSide) IsValid() bool {
	if s == Left || s == Right || s == Both {
		return true
	}
	return false
}
func SideByte(b byte) MatSide {
	switch b {
	case 'l':
		return Left
	case 'L':
		return Left
	case 'r':
		return Right
	case 'R':
		return Right
	case 'b':
		return Both
	case 'B':
		return Both
	default:
		return -1
	}
}
func IterMatSide() []MatSide {
	return []MatSide{Left, Right}
}

type MatOpts struct {
	Style   MatStyle
	Storage MatStorage
	Kl, Ku  int
	Uplo    MatUplo
	Diag    MatDiag
	Side    MatSide
	Major   MatMajor
}

func NewMatOpts() MatOpts {
	return MatOpts{
		Style:   General,
		Storage: Dense,
		Uplo:    Full,
		Diag:    NonUnit,
		Side:    Left,
		Major:   Row,
	}
}
func NewMatOptsCol() MatOpts {
	return MatOpts{
		Style:   General,
		Storage: Dense,
		Uplo:    Full,
		Diag:    NonUnit,
		Side:    Left,
		Major:   Col,
	}
}
func (m MatOpts) DeepCopy() MatOpts {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := MatOpts{}
	_ = d.Decode(&result)
	return result
}
func (m MatOpts) Iter() map[string]interface{} {
	opts := make(map[string]interface{})
	opts["style"] = m.Style
	opts["storage"] = m.Storage
	opts["kl"] = m.Kl
	opts["ku"] = m.Ku
	opts["uplo"] = m.Uplo
	opts["diag"] = m.Diag
	opts["side"] = m.Side
	opts["major"] = m.Major
	return opts
}

type TransBuilder struct {
	rows, cols int
	data       []float64
}

func NewTransBuilder(r, c int, x []float64) *TransBuilder {
	return &TransBuilder{rows: r, cols: c, data: x}
}
func (t *TransBuilder) T() []float64 {
	newmat := make([]float64, len(t.data))
	for i := 0; i < t.rows; i++ {
		for j := 0; j < t.cols; j++ {
			newmat[j+t.cols*i] = t.data[i+t.rows*j]
		}
	}
	copy(t.data, newmat)
	return t.data
}

type Matrix struct {
	Rows, Cols int
	Data       []float64
	Opts       MatOpts
}

func (m *Matrix) AppendCol(x []float64) {
	m.Cols += 1
	newData := make([]float64, m.Rows*m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			newData[getIdx(m.Opts.Major, m.Rows, m.Cols, i, j)] = m.Get(i, j)
		}
	}
	m.Data = newData
	for i := 0; i < m.Rows; i++ {
		m.Set(i, m.Cols-1, x[i])
	}
}
func (m *Matrix) AppendRow(x []float64) {
	m.Rows += 1
	newData := make([]float64, m.Rows*m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			newData[getIdx(m.Opts.Major, m.Rows, m.Cols, i, j)] = m.Get(i, j)
		}
	}
	m.Data = newData
	for i := 0; i < m.Cols; i++ {
		m.Set(m.Rows-1, i, x[i])
	}
}
func (m *Matrix) Get(r, c int) float64 {
	rnew, cnew := r, c

	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("Matrix.Get: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, rnew, cnew)]
}
func (m *Matrix) GetPtr(r, c int) *float64 {
	rnew, cnew := r, c

	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("Matrix.GetPtr: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return &m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, rnew, cnew)]
}
func (m *Matrix) GetIdx(idx int) float64 {
	return m.Data[idx]
}
func (m *Matrix) GetIdxPtr(idx int) *float64 {
	return &m.Data[idx]
}
func (m *Matrix) GetCmplx(r, c int) complex128 {
	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("Matrix.GetCmplx: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return complex(m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, r, c)], 0)
}
func (m *Matrix) GetMag(r, c int) float64 {
	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("Matrix.GetMag: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return math.Abs(m.Get(r, c))
}
func (m *Matrix) GetCmplxIdx(idx int) complex128 {
	return complex(m.Data[idx], 0)
}
func (m *Matrix) GetMagIdx(idx int) float64 {
	return math.Abs(m.Data[idx])
}
func (m *Matrix) Set(r, c int, x float64) {
	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("Matrix.Set: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, r, c)] = x
}
func (m *Matrix) SetAll(x float64) {
	for i := range m.Data {
		m.Data[i] = x
	}
}
func (m *Matrix) SetCol(c int, x float64) {
	for i := 0; i < m.Rows; i++ {
		m.Set(i, c, x)
	}
}
func (m *Matrix) SetRow(r int, x float64) {
	for i := 0; i < m.Cols; i++ {
		m.Set(r, i, x)
	}
}
func (m *Matrix) SetIdx(idx int, x float64) {
	m.Data[idx] = x
}
func (m *Matrix) T() []float64 {
	t := NewTransBuilder(m.Rows, m.Cols, m.Data)
	return t.T()
}
func (m *Matrix) Copy(r, c int) *Matrix {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := Matrix{}
	_ = d.Decode(&result)
	result.Data = m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, r, c):]
	return &result
}

// CopyIdx copies Matrix starting at idx index to a new Matrix
func (m *Matrix) CopyIdx(idx int) *Matrix {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := Matrix{}
	_ = d.Decode(&result)
	result.Data = m.Data[idx:]
	return &result
}
func (m *Matrix) DeepCopy() *Matrix {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	if err := e.Encode(m); err != nil {
		panic(err)
	}
	d := gob.NewDecoder(&b)
	result := &Matrix{}
	if len(m.Data) == 0 {
		result.Data = make([]float64, 0)
	}
	if err := d.Decode(result); err != nil {
		panic(err)
	}
	return result
}
func (m *Matrix) Off(r, c int) *Matrix {
	if getIdx(m.Opts.Major, m.Rows, m.Cols, r, c) >= m.Rows*m.Cols {
		log.Panicf("\n\ntrying to refernce start value greater than total array size! r=%v, c=%v, rows=%v, cols=%v, got %v, have %v\n\n\n", r, c, m.Rows, m.Cols, getIdx(m.Opts.Major, m.Rows, m.Cols, r, c), m.Rows*m.Cols)
	}
	return &Matrix{Rows: m.Rows, Cols: m.Cols, Opts: m.Opts, Data: m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, r, c):]}
}
func (m *Matrix) OffIdx(idx int) *Matrix {
	return &Matrix{Rows: m.Rows, Cols: m.Cols, Opts: m.Opts, Data: m.Data[idx:]}
}
func (m *Matrix) Vector(r, c int, n ...int) *Vector {
	inc := 1
	if n != nil {
		inc = n[0]
	} else if m.Opts.Major == Row {
		inc = m.Cols
	} else {
		inc = m.Rows
	}
	idx := getIdx(m.Opts.Major, m.Rows, m.Cols, r, c)

	return &Vector{Size: len(m.Data[idx:]), Inc: inc, Data: m.Data[idx:]}
}
func (m *Matrix) VectorIdx(idx int, n ...int) *Vector {
	inc := 1
	if n != nil {
		inc = n[0]
	} else if m.Opts.Major == Row {
		inc = m.Cols
	} else {
		inc = m.Rows
	}
	return &Vector{Size: len(m.Data), Inc: inc, Data: m.Data[idx:]}
}
func (m *Matrix) UpdateSize(r, c int) *Matrix {
	m.Rows, m.Cols = r, c
	return m
}
func (m *Matrix) UpdateCols(c int) *Matrix {
	m.Cols = c
	m.Rows = len(m.Data) / m.Cols
	return m
}
func (m *Matrix) UpdateRows(r int) *Matrix {
	m.Rows = r
	m.Cols = len(m.Data) / m.Rows
	return m
}
func (m *Matrix) ToColMajor() {
	if m.Opts.Major == Col {
		return
	}

	optsNew := m.Opts.DeepCopy()
	optsNew.Major = Col
	a := &Matrix{Rows: m.Rows, Cols: m.Cols, Opts: optsNew, Data: newMatrixData(m.Rows, m.Cols)}
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			a.Set(i, j, m.Get(i, j))
		}
	}
	m.Opts.Major = Col
	m.Data = a.Data
}
func (m *Matrix) ToRowMajor() {
	if m.Opts.Major == Row {
		return
	}

	optsNew := m.Opts.DeepCopy()
	optsNew.Major = Row
	a := &Matrix{Rows: m.Rows, Cols: m.Cols, Opts: optsNew, Data: newMatrixData(m.Rows, m.Cols)}
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			a.Set(i, j, m.Get(i, j))
		}
	}
	m.Opts.Major = Row
	m.Data = a.Data
}

func NewMatrix(r, c int, opts ...MatOpts) *Matrix {
	var o MatOpts

	if opts != nil {
		o = opts[0]
	}
	return &Matrix{Rows: r, Cols: c, Opts: o, Data: newMatrixData(r, c)}
}

func MatrixFactory() func(int, int, ...MatOpts) *Matrix {
	return func(r, c int, opts ...MatOpts) *Matrix {
		return NewMatrix(r, c, opts...)
	}
}

func MatrixDataFactory() func(*Matrix, []float64) *Matrix {
	return func(m *Matrix, d []float64) *Matrix {
		out := m.DeepCopy()
		out.Data = d
		return out
	}
}

func newMatrixData(r, c int) []float64 {
	return make([]float64, r*c)
}

type CTransBuilder struct {
	rows, cols int
	data       []complex128
}

func NewCTransBuilder(r, c int, x []complex128) *CTransBuilder {
	return &CTransBuilder{rows: r, cols: c, data: x}
}
func (t *CTransBuilder) T() []complex128 {
	newmat := make([]complex128, len(t.data))
	for i := 0; i < t.rows; i++ {
		for j := 0; j < t.cols; j++ {
			newmat[j+t.cols*i] = t.data[i+t.rows*j]
		}
	}
	copy(t.data, newmat)
	return t.data
}

type CMatrix struct {
	Rows, Cols int
	Data       []complex128
	Opts       MatOpts
}

func (m *CMatrix) AppendCol(x []complex128) {
	m.Cols += 1
	newData := make([]complex128, m.Rows*m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			newData[getIdx(m.Opts.Major, m.Rows, m.Cols, i, j)] = m.Get(i, j)
		}
	}
	m.Data = newData
	for i := 0; i < m.Rows; i++ {
		m.Set(i, m.Cols-1, x[i])
	}
}
func (m *CMatrix) AppendRow(x []complex128) {
	m.Rows += 1
	newData := make([]complex128, m.Rows*m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			newData[getIdx(m.Opts.Major, m.Rows, m.Cols, i, j)] = m.Get(i, j)
		}
	}
	m.Data = newData
	for i := 0; i < m.Cols; i++ {
		m.Set(m.Rows-1, i, x[i])
	}
}
func (m *CMatrix) Get(r, c int) complex128 {
	rnew, cnew := r, c

	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		// 		// log.Fatalf("MatrixDense.Get: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.rows, m.cols)
		log.Panicf("MatrixDense.Get: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, rnew, cnew)]
}
func (m *CMatrix) GetConj(r, c int) complex128 {
	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.GetConj: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return cmplx.Conj(m.Get(r, c))
}
func (m *CMatrix) GetMag(r, c int) float64 {
	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.GetMag: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return cmplx.Abs(m.Get(r, c))
}
func (m *CMatrix) GetArg(r, c int) float64 {
	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.GetArg: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return cmplx.Phase(m.Get(r, c))
}
func (m *CMatrix) GetDeg(r, c int) float64 {
	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.GetDeg: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return cmplx.Phase(m.Get(r, c)) * 180 / math.Pi
}
func (m *CMatrix) GetRe(r, c int) float64 {
	rnew, cnew := r, c

	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.GetRe: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return real(m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, rnew, cnew)])
}
func (m *CMatrix) GetIm(r, c int) float64 {
	rnew, cnew := r, c

	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.GetIm: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return imag(m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, rnew, cnew)])
}
func (m *CMatrix) GetReCmplx(r, c int) complex128 {
	rnew, cnew := r, c

	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.GetReCmplx: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return complex(real(m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, rnew, cnew)]), 0)
}
func (m *CMatrix) GetImCmplx(r, c int) complex128 {
	rnew, cnew := r, c

	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.GetImCmplx: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return complex(0, imag(m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, rnew, cnew)]))
}
func (m *CMatrix) GetConjProd(r, c int) float64 {
	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.GetConjProd: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return real(cmplx.Conj(m.Get(r, c)) * m.Get(r, c))
}
func (m *CMatrix) GetPtr(r, c int) *complex128 {
	rnew, cnew := r, c

	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.GetPtr: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return &m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, rnew, cnew)]
}
func (m *CMatrix) GetIdx(idx int) complex128 {
	return m.Data[idx]
}
func (m *CMatrix) GetConjIdx(idx int) complex128 {
	return cmplx.Conj(m.GetIdx(idx))
}
func (m *CMatrix) GetMagIdx(idx int) float64 {
	return cmplx.Abs(m.GetIdx(idx))
}
func (m *CMatrix) GetArgIdx(idx int) float64 {
	return cmplx.Phase(m.GetIdx(idx))
}
func (m *CMatrix) GetDegIdx(idx int) float64 {
	return cmplx.Phase(m.GetIdx(idx)) * 180 / math.Pi
}
func (m *CMatrix) GetReIdx(idx int) complex128 {
	return complex(real(m.GetIdx(idx)), 0)
}
func (m *CMatrix) GetImIdx(idx int) complex128 {
	return complex(0, imag(m.GetIdx(idx)))
}
func (m *CMatrix) GetConjProdIdx(idx int) float64 {
	return real(cmplx.Conj(m.GetIdx(idx)) * m.GetIdx(idx))
}
func (m *CMatrix) GetIdxPtr(idx int) *complex128 {
	return &m.Data[idx]
}
func (m *CMatrix) Set(r, c int, x complex128) {
	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.Set: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, r, c)] = x
}
func (m *CMatrix) SetAll(x complex128) {
	for i := range m.Data {
		m.Data[i] = x
	}
}
func (m *CMatrix) SetCol(c int, x complex128) {
	for i := 0; i < m.Rows; i++ {
		m.Set(i, c, x)
	}
}
func (m *CMatrix) SetRow(r int, x complex128) {
	for i := 0; i < m.Cols; i++ {
		m.Set(r, i, x)
	}
}
func (m *CMatrix) SetRe(r, c int, x float64) {
	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.SetRe: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, r, c)] = complex(x, 0)
}
func (m *CMatrix) SetIm(r, c int, x float64) {
	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.SetIm: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, r, c)] = complex(0, x)
}
func (m *CMatrix) SetReAll(x float64) {
	for i := range m.Data {
		m.Data[i] = complex(x, 0)
	}
}
func (m *CMatrix) SetReCol(c int, x float64) {
	for i := 0; i < m.Rows; i++ {
		m.SetRe(i, c, x)
	}
}
func (m *CMatrix) SetReRow(r int, x float64) {
	for i := 0; i < m.Cols; i++ {
		m.SetRe(r, i, x)
	}
}
func (m *CMatrix) SetImAll(x float64) {
	for i := range m.Data {
		m.Data[i] = complex(0, x)
	}
}
func (m *CMatrix) SetImCol(c int, x float64) {
	for i := 0; i < m.Rows; i++ {
		m.SetIm(i, c, x)
	}
}
func (m *CMatrix) SetImRow(r int, x float64) {
	for i := 0; i < m.Cols; i++ {
		m.SetIm(r, i, x)
	}
}
func (m *CMatrix) SetIdx(idx int, x complex128) {
	m.Data[idx] = x
}
func (m *CMatrix) T() []complex128 {
	t := NewCTransBuilder(m.Rows, m.Cols, m.Data)
	return t.T()
}
func (m *CMatrix) Copy(r, c int) *CMatrix {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := CMatrix{}
	_ = d.Decode(&result)
	result.Data = m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, r, c):]
	return &result
}
func (m *CMatrix) CopyIdx(idx int) *CMatrix {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := CMatrix{}
	_ = d.Decode(&result)
	result.Data = m.Data[idx:]
	return &result
}
func (m *CMatrix) DeepCopy() *CMatrix {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	if err := e.Encode(m); err != nil {
		panic(err)
	}
	d := gob.NewDecoder(&b)
	result := &CMatrix{}
	if len(m.Data) == 0 {
		result.Data = make([]complex128, 0)
	}
	if err := d.Decode(result); err != nil {
		panic(err)
	}
	return result
}
func (m *CMatrix) Off(r, c int) *CMatrix {
	if getIdx(m.Opts.Major, m.Rows, m.Cols, r, c) >= m.Rows*m.Cols {
		log.Panicf("\n\ntrying to refernce start value greater than total array size! r=%v, c=%v, rows=%v, cols=%v, got %v, have %v\n\n\n", r, c, m.Rows, m.Cols, getIdx(m.Opts.Major, m.Rows, m.Cols, r, c), m.Rows*m.Cols)
	}
	return &CMatrix{Rows: m.Rows, Cols: m.Cols, Opts: m.Opts, Data: m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, r, c):]}
}
func (m *CMatrix) OffIdx(idx int) *CMatrix {
	return &CMatrix{Rows: m.Rows, Cols: m.Cols, Opts: m.Opts, Data: m.Data[idx:]}
}
func (m *CMatrix) CVector(r, c int, n ...int) *CVector {
	inc := 1
	if n != nil {
		inc = n[0]
	} else if m.Opts.Major == Row {
		inc = m.Cols
	} else {
		inc = m.Rows
	}
	idx := getIdx(m.Opts.Major, m.Rows, m.Cols, r, c)

	return &CVector{Size: len(m.Data[idx:]), Inc: inc, Data: m.Data[idx:]}
}
func (m *CMatrix) CVectorIdx(idx int, n ...int) *CVector {
	inc := 1
	if n != nil {
		inc = n[0]
	} else if m.Opts.Major == Row {
		inc = m.Cols
	} else {
		inc = m.Rows
	}
	return &CVector{Size: len(m.Data), Inc: inc, Data: m.Data[idx:]}
}
func (m *CMatrix) UpdateSize(r, c int) *CMatrix {
	m.Rows, m.Cols = r, c
	return m
}
func (m *CMatrix) UpdateCols(c int) *CMatrix {
	m.Cols = c
	m.Rows = len(m.Data) / m.Cols
	return m
}
func (m *CMatrix) UpdateRows(r int) *CMatrix {
	m.Rows = r
	m.Cols = len(m.Data) / m.Rows
	return m
}
func (m *CMatrix) ToColMajor() {
	if m.Opts.Major == Col {
		return
	}

	optsNew := m.Opts.DeepCopy()
	optsNew.Major = Col
	a := &CMatrix{Rows: m.Rows, Cols: m.Cols, Opts: optsNew, Data: newCMatrixData(m.Rows, m.Cols)}
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			a.Set(i, j, m.Get(i, j))
		}
	}
	m.Opts.Major = Col
	m.Data = a.Data
}
func (m *CMatrix) ToRowMajor() {
	if m.Opts.Major == Row {
		return
	}

	optsNew := m.Opts.DeepCopy()
	optsNew.Major = Row
	a := &CMatrix{Rows: m.Rows, Cols: m.Cols, Opts: optsNew, Data: newCMatrixData(m.Rows, m.Cols)}
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			a.Set(i, j, m.Get(i, j))
		}
	}
	m.Opts.Major = Row
	m.Data = a.Data
}

func newCMatrixData(r, c int) []complex128 {
	return make([]complex128, r*c)
}

func NewCMatrix(r, c int, opts ...MatOpts) *CMatrix {
	var o MatOpts

	if opts != nil {
		o = opts[0]
	} else {
		o = NewMatOpts()
	}
	return &CMatrix{Rows: r, Cols: c, Opts: o, Data: newCMatrixData(r, c)}
}

func CMatrixFactory() func(int, int, ...MatOpts) *CMatrix {
	return func(r, c int, opts ...MatOpts) *CMatrix {
		return NewCMatrix(r, c, opts...)
	}
}

func CMatrixDataFactory() func(*CMatrix, []complex128) *CMatrix {
	return func(m *CMatrix, d []complex128) *CMatrix {
		out := m.DeepCopy()
		out.Data = d
		return out
	}
}
