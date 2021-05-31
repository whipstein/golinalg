package lin

import (
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// Zerrlqtp tests the error exits for the complex routines
// that use the LQT decomposition of a triangular-pentagonal matrix.
func Zerrlqtp(path []byte, _t *testing.T) {
	var i, info, j, nmax int

	w := cvf(2)
	a := cmf(2, 2, opts)
	b := cmf(2, 2, opts)
	c := cmf(2, 2, opts)
	t := cmf(2, 2, opts)

	nmax = 2
	infot := &gltest.Common.Infoc.Infot
	ok := &gltest.Common.Infoc.Ok
	lerr := &gltest.Common.Infoc.Lerr
	srnamt := &gltest.Common.Srnamc.Srnamt

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./complex(float64(i+j), 0))
			c.Set(i-1, j-1, 1./complex(float64(i+j), 0))
			t.Set(i-1, j-1, 1./complex(float64(i+j), 0))
		}
		w.Set(j-1, 0.0)
	}
	(*ok) = true

	//     Error exits for TPLQT factorization
	//
	//     ZTPLQT
	*srnamt = "ZTPLQT"
	(*infot) = 1
	golapack.Ztplqt(toPtr(-1), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZTPLQT", &info, lerr, ok, _t)
	(*infot) = 2
	golapack.Ztplqt(func() *int { y := 1; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZTPLQT", &info, lerr, ok, _t)
	(*infot) = 3
	golapack.Ztplqt(func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), toPtr(-1), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZTPLQT", &info, lerr, ok, _t)
	(*infot) = 3
	golapack.Ztplqt(func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZTPLQT", &info, lerr, ok, _t)
	(*infot) = 4
	golapack.Ztplqt(func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZTPLQT", &info, lerr, ok, _t)
	(*infot) = 4
	golapack.Ztplqt(func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZTPLQT", &info, lerr, ok, _t)
	(*infot) = 6
	golapack.Ztplqt(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZTPLQT", &info, lerr, ok, _t)
	(*infot) = 8
	golapack.Ztplqt(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZTPLQT", &info, lerr, ok, _t)
	(*infot) = 10
	golapack.Ztplqt(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), t, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZTPLQT", &info, lerr, ok, _t)

	//     ZTPLQT2
	*srnamt = "ZTPLQT2"
	(*infot) = 1
	golapack.Ztplqt2(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZTPLQT2", &info, lerr, ok, _t)
	(*infot) = 2
	golapack.Ztplqt2(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZTPLQT2", &info, lerr, ok, _t)
	(*infot) = 3
	golapack.Ztplqt2(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZTPLQT2", &info, lerr, ok, _t)
	(*infot) = 5
	golapack.Ztplqt2(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, func() *int { y := 2; return &y }(), t, func() *int { y := 2; return &y }(), &info)
	Chkxer("ZTPLQT2", &info, lerr, ok, _t)
	(*infot) = 7
	golapack.Ztplqt2(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 1; return &y }(), t, func() *int { y := 2; return &y }(), &info)
	Chkxer("ZTPLQT2", &info, lerr, ok, _t)
	(*infot) = 9
	golapack.Ztplqt2(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), b, func() *int { y := 2; return &y }(), t, func() *int { y := 1; return &y }(), &info)
	Chkxer("ZTPLQT2", &info, lerr, ok, _t)

	//     ZTPMLQT
	*srnamt = "ZTPMLQT"
	(*infot) = 1
	golapack.Ztpmlqt('/', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZTPMLQT", &info, lerr, ok, _t)
	(*infot) = 2
	golapack.Ztpmlqt('L', '/', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZTPMLQT", &info, lerr, ok, _t)
	(*infot) = 3
	golapack.Ztpmlqt('L', 'N', toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZTPMLQT", &info, lerr, ok, _t)
	(*infot) = 4
	golapack.Ztpmlqt('L', 'N', func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZTPMLQT", &info, lerr, ok, _t)
	(*infot) = 5
	golapack.Ztpmlqt('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	(*infot) = 6
	golapack.Ztpmlqt('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZTPMLQT", &info, lerr, ok, _t)
	(*infot) = 7
	golapack.Ztpmlqt('L', 'N', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZTPMLQT", &info, lerr, ok, _t)
	(*infot) = 9
	golapack.Ztpmlqt('R', 'N', func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZTPMLQT", &info, lerr, ok, _t)
	(*infot) = 11
	golapack.Ztpmlqt('R', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 0; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZTPMLQT", &info, lerr, ok, _t)
	(*infot) = 13
	golapack.Ztpmlqt('L', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), b, func() *int { y := 0; return &y }(), c, func() *int { y := 1; return &y }(), w, &info)
	Chkxer("ZTPMLQT", &info, lerr, ok, _t)
	(*infot) = 15
	golapack.Ztpmlqt('L', 'N', func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), t, func() *int { y := 1; return &y }(), b, func() *int { y := 1; return &y }(), c, func() *int { y := 0; return &y }(), w, &info)
	Chkxer("ZTPMLQT", &info, lerr, ok, _t)

	//     Print a summary line.
	Alaesm(path, ok)
}
