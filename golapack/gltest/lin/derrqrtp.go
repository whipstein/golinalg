package lin

import (
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// Derrqrtp tests the error exits for the REAL routines
// that use the QRT decomposition of a triangular-pentagonal matrix.
func Derrqrtp(path []byte, _t *testing.T) {
	var i, info, j, nmax int
	lerr := &gltest.Common.Infoc.Lerr
	ok := &gltest.Common.Infoc.Ok
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	nmax = 2

	a := mf(2, 2, opts)
	b := mf(2, 2, opts)
	c := mf(2, 2, opts)
	t := mf(2, 2, opts)
	w := vf(2)

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./float64(i+j))
			c.Set(i-1, j-1, 1./float64(i+j))
			t.Set(i-1, j-1, 1./float64(i+j))
		}
		w.Set(j-1, 0.0)
	}
	(*ok) = true

	//     Error exits for TPQRT factorization
	//
	//     DTPQRT
	*srnamt = "DTPQRT"
	*infot = 1
	golapack.Dtpqrt(toPtr(-1), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DTPQRT", &info, lerr, ok, _t)
	*infot = 2
	golapack.Dtpqrt(func() *int { y := 1; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DTPQRT", &info, lerr, ok, _t)
	*infot = 3
	golapack.Dtpqrt(func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), toPtr(-1), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DTPQRT", &info, lerr, ok, _t)
	*infot = 3
	golapack.Dtpqrt(func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DTPQRT", &info, lerr, ok, _t)
	*infot = 4
	golapack.Dtpqrt(func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DTPQRT", &info, lerr, ok, _t)
	*infot = 4
	golapack.Dtpqrt(func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DTPQRT", &info, lerr, ok, _t)
	*infot = 6
	golapack.Dtpqrt(func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DTPQRT", &info, lerr, ok, _t)
	*infot = 8
	golapack.Dtpqrt(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DTPQRT", &info, lerr, ok, _t)
	*infot = 10
	golapack.Dtpqrt(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DTPQRT", &info, lerr, ok, _t)

	//     DTPQRT2
	*srnamt = "DTPQRT2"
	*infot = 1
	golapack.Dtpqrt2(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), &info)
	Chkxer("DTPQRT2", &info, lerr, ok, _t)
	*infot = 2
	golapack.Dtpqrt2(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), &info)
	Chkxer("DTPQRT2", &info, lerr, ok, _t)
	*infot = 3
	golapack.Dtpqrt2(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), &info)
	Chkxer("DTPQRT2", &info, lerr, ok, _t)
	*infot = 5
	golapack.Dtpqrt2(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), t, func() *int { y := 2; return &y }(), &info)
	Chkxer("DTPQRT2", &info, lerr, ok, _t)
	*infot = 7
	golapack.Dtpqrt2(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), t, func() *int { y := 2; return &y }(), &info)
	Chkxer("DTPQRT2", &info, lerr, ok, _t)
	*infot = 9
	golapack.Dtpqrt2(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), t, func() *int { y := 1; return &y }(), &info)
	Chkxer("DTPQRT2", &info, lerr, ok, _t)

	//     DTPMQRT
	*srnamt = "DTPMQRT"
	*infot = 1
	golapack.Dtpmqrt('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DTPMQRT", &info, lerr, ok, _t)
	*infot = 2
	golapack.Dtpmqrt('L', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DTPMQRT", &info, lerr, ok, _t)
	*infot = 3
	golapack.Dtpmqrt('L', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DTPMQRT", &info, lerr, ok, _t)
	*infot = 4
	golapack.Dtpmqrt('L', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DTPMQRT", &info, lerr, ok, _t)
	*infot = 5
	golapack.Dtpmqrt('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	*infot = 6
	golapack.Dtpmqrt('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DTPMQRT", &info, lerr, ok, _t)
	*infot = 7
	golapack.Dtpmqrt('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DTPMQRT", &info, lerr, ok, _t)
	*infot = 9
	golapack.Dtpmqrt('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DTPMQRT", &info, lerr, ok, _t)
	*infot = 9
	golapack.Dtpmqrt('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DTPMQRT", &info, lerr, ok, _t)
	*infot = 11
	golapack.Dtpmqrt('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DTPMQRT", &info, lerr, ok, _t)
	*infot = 13
	golapack.Dtpmqrt('L', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DTPMQRT", &info, lerr, ok, _t)
	*infot = 15
	golapack.Dtpmqrt('L', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 0; return &y }(), w, &info)
	Chkxer("DTPMQRT", &info, lerr, ok, _t)

	//     Print a summary line.
	Alaesm(path, ok)
}
