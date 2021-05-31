package lin

import (
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// Derrqrt tests the error exits for the DOUBLE PRECISION routines
// that use the QRT decomposition of a general matrix.
func Derrqrt(path []byte, t *testing.T) {
	var i, info, j, nmax int
	lerr := &gltest.Common.Infoc.Lerr
	ok := &gltest.Common.Infoc.Ok
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	nmax = 2

	a := mf(2, 2, opts)
	c := mf(2, 2, opts)
	_t := mf(2, 2, opts)
	w := vf(2)

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./float64(i+j))
			c.Set(i-1, j-1, 1./float64(i+j))
			_t.Set(i-1, j-1, 1./float64(i+j))
		}
		w.Set(j-1, 0.)
	}
	(*ok) = true

	//     Error exits for QRT factorization
	//
	//     DGEQRT
	*srnamt = "DGEQRT"
	*infot = 1
	golapack.Dgeqrt(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), _t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEQRT", &info, lerr, ok, t)
	*infot = 2
	golapack.Dgeqrt(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), _t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEQRT", &info, lerr, ok, t)
	*infot = 3
	golapack.Dgeqrt(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), _t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEQRT", &info, lerr, ok, t)
	*infot = 5
	golapack.Dgeqrt(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), _t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEQRT", &info, lerr, ok, t)
	*infot = 7
	golapack.Dgeqrt(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), _t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEQRT", &info, lerr, ok, t)

	//     DGEQRT2
	*srnamt = "DGEQRT2"
	*infot = 1
	golapack.Dgeqrt2(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), _t, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEQRT2", &info, lerr, ok, t)
	*infot = 2
	golapack.Dgeqrt2(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), _t, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEQRT2", &info, lerr, ok, t)
	*infot = 4
	golapack.Dgeqrt2(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), _t, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEQRT2", &info, lerr, ok, t)
	*infot = 6
	golapack.Dgeqrt2(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), _t, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEQRT2", &info, lerr, ok, t)

	//     DGEQRT3
	*srnamt = "DGEQRT3"
	*infot = 1
	golapack.Dgeqrt3(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), _t, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEQRT3", &info, lerr, ok, t)
	*infot = 2
	golapack.Dgeqrt3(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), _t, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEQRT3", &info, lerr, ok, t)
	*infot = 4
	golapack.Dgeqrt3(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), _t, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEQRT3", &info, lerr, ok, t)
	*infot = 6
	golapack.Dgeqrt3(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), _t, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGEQRT3", &info, lerr, ok, t)

	//     DGEMQRT
	*srnamt = "DGEMQRT"
	*infot = 1
	golapack.Dgemqrt('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), _t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEMQRT", &info, lerr, ok, t)
	*infot = 2
	golapack.Dgemqrt('L', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), _t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEMQRT", &info, lerr, ok, t)
	*infot = 3
	golapack.Dgemqrt('L', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), _t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEMQRT", &info, lerr, ok, t)
	*infot = 4
	golapack.Dgemqrt('L', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), _t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEMQRT", &info, lerr, ok, t)
	*infot = 5
	golapack.Dgemqrt('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), _t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEMQRT", &info, lerr, ok, t)
	*infot = 5
	golapack.Dgemqrt('R', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), _t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEMQRT", &info, lerr, ok, t)
	*infot = 6
	golapack.Dgemqrt('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), _t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEMQRT", &info, lerr, ok, t)
	*infot = 8
	golapack.Dgemqrt('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), _t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEMQRT", &info, lerr, ok, t)
	*infot = 8
	golapack.Dgemqrt('L', 'N', func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), _t, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEMQRT", &info, lerr, ok, t)
	*infot = 10
	golapack.Dgemqrt('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), _t, func() *int { y := 0; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("DGEMQRT", &info, lerr, ok, t)
	*infot = 12
	golapack.Dgemqrt('L', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), _t, func() *int { y := 1; return &y }(), c, func() *int { y := 0; return &y }(), w, &info)
	Chkxer("DGEMQRT", &info, lerr, ok, t)

	//     Print a summary line.
	Alaesm(path, ok)
}
