package lin

import (
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// Zerrunhrcol tests the error exits for ZUNHR_COL that does
// Householder reconstruction from the ouput of tall-skinny
// factorization ZLATSQR.
func Zerrunhrcol(path []byte, _t *testing.T) {
	var i, info, j, nmax int

	d := cvf(2)
	a := cmf(2, 2, opts)
	t := cmf(2, 2, opts)

	nmax = 2
	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	lerr := &gltest.Common.Infoc.Lerr
	srnamt := &gltest.Common.Srnamc.Srnamt

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.SetRe(i-1, j-1, 1./float64(i+j))
			t.SetRe(i-1, j-1, 1./float64(i+j))
		}
		d.Set(j-1, (0. + 0.*1i))
	}
	(*ok) = true

	//     Error exits for Householder reconstruction
	//
	//     ZUNHR_COL
	*srnamt = "ZUNHR_COL"

	*infot = 1
	golapack.Zunhrcol(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), d, &info)
	Chkxer("ZUNHR_COL", &info, lerr, ok, _t)

	*infot = 2
	golapack.Zunhrcol(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), d, &info)
	Chkxer("ZUNHR_COL", &info, lerr, ok, _t)
	golapack.Zunhrcol(func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), d, &info)
	Chkxer("ZUNHR_COL", &info, lerr, ok, _t)

	*infot = 3
	golapack.Zunhrcol(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), d, &info)
	Chkxer("ZUNHR_COL", &info, lerr, ok, _t)

	golapack.Zunhrcol(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), d, &info)
	Chkxer("ZUNHR_COL", &info, lerr, ok, _t)

	*infot = 5
	golapack.Zunhrcol(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, toPtr(-1), t, func() *int { y := 1; return &y }(), d, &info)
	Chkxer("ZUNHR_COL", &info, lerr, ok, _t)

	golapack.Zunhrcol(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 0; return &y }(), t, func() *int { y := 1; return &y }(), d, &info)
	Chkxer("ZUNHR_COL", &info, lerr, ok, _t)

	golapack.Zunhrcol(func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), d, &info)
	Chkxer("ZUNHR_COL", &info, lerr, ok, _t)

	*infot = 7
	golapack.Zunhrcol(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, toPtr(-1), d, &info)
	Chkxer("ZUNHR_COL", &info, lerr, ok, _t)

	golapack.Zunhrcol(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 0; return &y }(), d, &info)
	Chkxer("ZUNHR_COL", &info, lerr, ok, _t)

	golapack.Zunhrcol(func() *int { y := 4; return &y }(), func() *int { y := 3; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 4; return &y }(), t, func() *int { y := 1; return &y }(), d, &info)
	Chkxer("ZUNHR_COL", &info, lerr, ok, _t)

	//     Print a summary line.
	Alaesm(path, ok)
}
