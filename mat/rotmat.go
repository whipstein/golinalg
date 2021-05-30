package mat

type DrotMatrix struct {
	Flag               int
	H11, H21, H12, H22 float64
}

func NewDrotMatrix() *DrotMatrix {
	return &DrotMatrix{Flag: 0, H11: 0, H21: 0, H12: 0, H22: 0}
}

func (d *DrotMatrix) H() [4]float64 {
	return [4]float64{d.H11, d.H21, d.H12, d.H22}
}

type DrotMatrixBuilder struct {
	matrix *DrotMatrix
}

func NewDrotMatrixBuilder() *DrotMatrixBuilder {
	return &DrotMatrixBuilder{
		matrix: &DrotMatrix{},
	}
}

func CreateDrotMatrixBuilder(dm *DrotMatrix) *DrotMatrixBuilder {
	return &DrotMatrixBuilder{
		matrix: dm,
	}
}

func (dm *DrotMatrixBuilder) Flag(f int) *DrotMatrixBuilder {
	dm.matrix.Flag = f
	return dm
}

func (dm *DrotMatrixBuilder) H(h [4]float64) *DrotMatrixBuilder {
	dm.matrix.H11 = h[0]
	dm.matrix.H21 = h[1]
	dm.matrix.H12 = h[2]
	dm.matrix.H22 = h[3]
	return dm
}

func (dm *DrotMatrixBuilder) Build() *DrotMatrix {
	return dm.matrix
}
