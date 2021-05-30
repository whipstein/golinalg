package lin

import (
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Derrorhrcol tests the error exits for DORHR_COL that does
// Householder reconstruction from the ouput of tall-skinny
// factorization DLATSQR.
func DerrorhrCol(path []byte, _t *testing.T) {
	var i, info, j, nmax int
	lerr := &gltest.Common.Infoc.Lerr
	ok := &gltest.Common.Infoc.Ok
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	nmax = 2

	a := mf(2, 2, opts)
	t := mf(2, 2, opts)
	d := vf(2)

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./float64(i+j))
			t.Set(i-1, j-1, 1./float64(i+j))
		}
		d.Set(j-1, 0.)
	}
	(*ok) = true

	//     Error exits for Householder reconstruction
	//
	//     DORHR_COL
	*srnamt = "DORHR_COL"
	*infot = 1
	golapack.DorhrCol(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), d, &info)
	Chkxer("DORHR_COL", &info, lerr, ok, _t)
	*infot = 2
	golapack.DorhrCol(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), d, &info)
	Chkxer("DORHR_COL", &info, lerr, ok, _t)
	golapack.DorhrCol(func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), d, &info)
	Chkxer("DORHR_COL", &info, lerr, ok, _t)
	*infot = 3
	golapack.DorhrCol(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), d, &info)
	Chkxer("DORHR_COL", &info, lerr, ok, _t)
	golapack.DorhrCol(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), d, &info)
	Chkxer("DORHR_COL", &info, lerr, ok, _t)
	*infot = 5
	golapack.DorhrCol(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, toPtr(-1), t, func() *int { y := 1; return &y }(), d, &info)
	Chkxer("DORHR_COL", &info, lerr, ok, _t)
	golapack.DorhrCol(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), t, func() *int { y := 1; return &y }(), d, &info)
	Chkxer("DORHR_COL", &info, lerr, ok, _t)
	golapack.DorhrCol(func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), d, &info)
	Chkxer("DORHR_COL", &info, lerr, ok, _t)
	*infot = 7
	golapack.DorhrCol(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, toPtr(-1), d, &info)
	Chkxer("DORHR_COL", &info, lerr, ok, _t)
	golapack.DorhrCol(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 0; return &y }(), d, &info)
	Chkxer("DORHR_COL", &info, lerr, ok, _t)
	golapack.DorhrCol(func() *int { y := 4; return &y }(), func() *int { y := 3; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 4; return &y }(), t, func() *int { y := 1; return &y }(), d, &info)
	Chkxer("DORHR_COL", &info, lerr, ok, _t)

	//     Print a summary line.
	Alaesm(path, ok)
}
