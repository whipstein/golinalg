package mat

import "testing"

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
