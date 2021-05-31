package lin

import (
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
)

// Derrqr tests the error exits for the DOUBLE PRECISION routines
// that use the QR decomposition of a general matrix.
func Derrqr(path []byte, t *testing.T) {
	var i, info, j, nmax int
	lerr := &gltest.Common.Infoc.Lerr
	ok := &gltest.Common.Infoc.Ok
	infot := &gltest.Common.Infoc.Infot
	srnamt := &gltest.Common.Srnamc.Srnamt

	nmax = 2

	a := mf(2, 2, opts)
	af := mf(2, 2, opts)
	b := vf(2)
	w := vf(2)
	x := vf(2)

	//     Set the variables to innocuous values.
	for j = 1; j <= nmax; j++ {
		for i = 1; i <= nmax; i++ {
			a.Set(i-1, j-1, 1./float64(i+j))
			af.Set(i-1, j-1, 1./float64(i+j))
		}
		b.Set(j-1, 0.)
		w.Set(j-1, 0.)
		x.Set(j-1, 0.)
	}
	(*ok) = true

	//     Error exits for QR factorization
	//     DGEQRF
	*srnamt = "DGEQRF"
	*infot = 1
	golapack.Dgeqrf(toPtr(-1), toPtr(0), a, toPtr(1), b, w, toPtr(1), &info)
	Chkxer("DGEQRF", &info, lerr, ok, t)
	*infot = 2
	golapack.Dgeqrf(toPtr(0), toPtr(-1), a, toPtr(1), b, w, toPtr(1), &info)
	Chkxer("DGEQRF", &info, lerr, ok, t)
	*infot = 4
	golapack.Dgeqrf(toPtr(2), toPtr(1), a, toPtr(1), b, w, toPtr(1), &info)
	Chkxer("DGEQRF", &info, lerr, ok, t)
	*infot = 7
	golapack.Dgeqrf(toPtr(1), toPtr(2), a, toPtr(1), b, w, toPtr(1), &info)
	Chkxer("DGEQRF", &info, lerr, ok, t)

	//     DGEQRFP
	*srnamt = "DGEQRFP"
	*infot = 1
	golapack.Dgeqrfp(toPtr(-1), toPtr(0), a, toPtr(1), b, w, toPtr(1), &info)
	Chkxer("DGEQRFP", &info, lerr, ok, t)
	*infot = 2
	golapack.Dgeqrfp(toPtr(0), toPtr(-1), a, toPtr(1), b, w, toPtr(1), &info)
	Chkxer("DGEQRFP", &info, lerr, ok, t)
	*infot = 4
	golapack.Dgeqrfp(toPtr(2), toPtr(1), a, toPtr(1), b, w, toPtr(1), &info)
	Chkxer("DGEQRFP", &info, lerr, ok, t)
	*infot = 7
	golapack.Dgeqrfp(toPtr(1), toPtr(2), a, toPtr(1), b, w, toPtr(1), &info)
	Chkxer("DGEQRFP", &info, lerr, ok, t)

	//     DGEQR2
	*srnamt = "DGEQR2"
	*infot = 1
	golapack.Dgeqr2(toPtr(-1), toPtr(0), a, toPtr(1), b, w, &info)
	Chkxer("DGEQR2", &info, lerr, ok, t)
	*infot = 2
	golapack.Dgeqr2(toPtr(0), toPtr(-1), a, toPtr(1), b, w, &info)
	Chkxer("DGEQR2", &info, lerr, ok, t)
	*infot = 4
	golapack.Dgeqr2(toPtr(2), toPtr(1), a, toPtr(1), b, w, &info)
	Chkxer("DGEQR2", &info, lerr, ok, t)

	//     DGEQR2P
	*srnamt = "DGEQR2P"
	*infot = 1
	golapack.Dgeqr2p(toPtr(-1), toPtr(0), a, toPtr(1), b, w, &info)
	Chkxer("DGEQR2P", &info, lerr, ok, t)
	*infot = 2
	golapack.Dgeqr2p(toPtr(0), toPtr(-1), a, toPtr(1), b, w, &info)
	Chkxer("DGEQR2P", &info, lerr, ok, t)
	*infot = 4
	golapack.Dgeqr2p(toPtr(2), toPtr(1), a, toPtr(1), b, w, &info)
	Chkxer("DGEQR2P", &info, lerr, ok, t)

	//     DGEQRS
	*srnamt = "DGEQRS"
	*infot = 1
	Dgeqrs(toPtr(-1), toPtr(0), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DGEQRS", &info, lerr, ok, t)
	*infot = 2
	Dgeqrs(toPtr(0), toPtr(-1), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DGEQRS", &info, lerr, ok, t)
	*infot = 2
	Dgeqrs(toPtr(1), toPtr(2), toPtr(0), a, toPtr(2), x, af, toPtr(2), w, toPtr(1), &info)
	Chkxer("DGEQRS", &info, lerr, ok, t)
	*infot = 3
	Dgeqrs(toPtr(0), toPtr(0), toPtr(-1), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DGEQRS", &info, lerr, ok, t)
	*infot = 5
	Dgeqrs(toPtr(2), toPtr(1), toPtr(0), a, toPtr(1), x, af, toPtr(2), w, toPtr(1), &info)
	Chkxer("DGEQRS", &info, lerr, ok, t)
	*infot = 8
	Dgeqrs(toPtr(2), toPtr(1), toPtr(0), a, toPtr(2), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DGEQRS", &info, lerr, ok, t)
	*infot = 10
	Dgeqrs(toPtr(1), toPtr(1), toPtr(2), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DGEQRS", &info, lerr, ok, t)

	//     DORGQR
	*srnamt = "DORGQR"
	*infot = 1
	golapack.Dorgqr(toPtr(-1), toPtr(0), toPtr(0), a, toPtr(1), x, w, toPtr(1), &info)
	Chkxer("DORGQR", &info, lerr, ok, t)
	*infot = 2
	golapack.Dorgqr(toPtr(0), toPtr(-1), toPtr(0), a, toPtr(1), x, w, toPtr(1), &info)
	Chkxer("DORGQR", &info, lerr, ok, t)
	*infot = 2
	golapack.Dorgqr(toPtr(1), toPtr(2), toPtr(0), a, toPtr(1), x, w, toPtr(2), &info)
	Chkxer("DORGQR", &info, lerr, ok, t)
	*infot = 3
	golapack.Dorgqr(toPtr(0), toPtr(0), toPtr(-1), a, toPtr(1), x, w, toPtr(1), &info)
	Chkxer("DORGQR", &info, lerr, ok, t)
	*infot = 3
	golapack.Dorgqr(toPtr(1), toPtr(1), toPtr(2), a, toPtr(1), x, w, toPtr(1), &info)
	Chkxer("DORGQR", &info, lerr, ok, t)
	*infot = 5
	golapack.Dorgqr(toPtr(2), toPtr(2), toPtr(0), a, toPtr(1), x, w, toPtr(2), &info)
	Chkxer("DORGQR", &info, lerr, ok, t)
	*infot = 8
	golapack.Dorgqr(toPtr(2), toPtr(2), toPtr(0), a, toPtr(2), x, w, toPtr(1), &info)
	Chkxer("DORGQR", &info, lerr, ok, t)

	//     DORG2R
	*srnamt = "DORG2R"
	*infot = 1
	golapack.Dorg2r(toPtr(-1), toPtr(0), toPtr(0), a, toPtr(1), x, w, &info)
	Chkxer("DORG2R", &info, lerr, ok, t)
	*infot = 2
	golapack.Dorg2r(toPtr(0), toPtr(-1), toPtr(0), a, toPtr(1), x, w, &info)
	Chkxer("DORG2R", &info, lerr, ok, t)
	*infot = 2
	golapack.Dorg2r(toPtr(1), toPtr(2), toPtr(0), a, toPtr(1), x, w, &info)
	Chkxer("DORG2R", &info, lerr, ok, t)
	*infot = 3
	golapack.Dorg2r(toPtr(0), toPtr(0), toPtr(-1), a, toPtr(1), x, w, &info)
	Chkxer("DORG2R", &info, lerr, ok, t)
	*infot = 3
	golapack.Dorg2r(toPtr(2), toPtr(1), toPtr(2), a, toPtr(2), x, w, &info)
	Chkxer("DORG2R", &info, lerr, ok, t)
	*infot = 5
	golapack.Dorg2r(toPtr(2), toPtr(1), toPtr(0), a, toPtr(1), x, w, &info)
	Chkxer("DORG2R", &info, lerr, ok, t)

	//     DORMQR
	*srnamt = "DORMQR"
	*infot = 1
	golapack.Dormqr('/', 'N', toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DORMQR", &info, lerr, ok, t)
	*infot = 2
	golapack.Dormqr('L', '/', toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DORMQR", &info, lerr, ok, t)
	*infot = 3
	golapack.Dormqr('L', 'N', toPtr(-1), toPtr(0), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DORMQR", &info, lerr, ok, t)
	*infot = 4
	golapack.Dormqr('L', 'N', toPtr(0), toPtr(-1), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DORMQR", &info, lerr, ok, t)
	*infot = 5
	golapack.Dormqr('L', 'N', toPtr(0), toPtr(0), toPtr(-1), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DORMQR", &info, lerr, ok, t)
	*infot = 5
	golapack.Dormqr('L', 'N', toPtr(0), toPtr(1), toPtr(1), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DORMQR", &info, lerr, ok, t)
	*infot = 5
	golapack.Dormqr('R', 'N', toPtr(1), toPtr(0), toPtr(1), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DORMQR", &info, lerr, ok, t)
	*infot = 7
	golapack.Dormqr('L', 'N', toPtr(2), toPtr(1), toPtr(0), a, toPtr(1), x, af, toPtr(2), w, toPtr(1), &info)
	Chkxer("DORMQR", &info, lerr, ok, t)
	*infot = 7
	golapack.Dormqr('R', 'N', toPtr(1), toPtr(2), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DORMQR", &info, lerr, ok, t)
	*infot = 10
	golapack.Dormqr('L', 'N', toPtr(2), toPtr(1), toPtr(0), a, toPtr(2), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DORMQR", &info, lerr, ok, t)
	*infot = 12
	golapack.Dormqr('L', 'N', toPtr(1), toPtr(2), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DORMQR", &info, lerr, ok, t)
	*infot = 12
	golapack.Dormqr('R', 'N', toPtr(2), toPtr(1), toPtr(0), a, toPtr(1), x, af, toPtr(2), w, toPtr(1), &info)
	Chkxer("DORMQR", &info, lerr, ok, t)

	//     DORM2R
	*srnamt = "DORM2R"
	*infot = 1
	golapack.Dorm2r('/', 'N', toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, &info)
	Chkxer("DORM2R", &info, lerr, ok, t)
	*infot = 2
	golapack.Dorm2r('L', '/', toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, &info)
	Chkxer("DORM2R", &info, lerr, ok, t)
	*infot = 3
	golapack.Dorm2r('L', 'N', toPtr(-1), toPtr(0), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, &info)
	Chkxer("DORM2R", &info, lerr, ok, t)
	*infot = 4
	golapack.Dorm2r('L', 'N', toPtr(0), toPtr(-1), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, &info)
	Chkxer("DORM2R", &info, lerr, ok, t)
	*infot = 5
	golapack.Dorm2r('L', 'N', toPtr(0), toPtr(0), toPtr(-1), a, toPtr(1), x, af, toPtr(1), w, &info)
	Chkxer("DORM2R", &info, lerr, ok, t)
	*infot = 5
	golapack.Dorm2r('L', 'N', toPtr(0), toPtr(1), toPtr(1), a, toPtr(1), x, af, toPtr(1), w, &info)
	Chkxer("DORM2R", &info, lerr, ok, t)
	*infot = 5
	golapack.Dorm2r('R', 'N', toPtr(1), toPtr(0), toPtr(1), a, toPtr(1), x, af, toPtr(1), w, &info)
	Chkxer("DORM2R", &info, lerr, ok, t)
	*infot = 7
	golapack.Dorm2r('L', 'N', toPtr(2), toPtr(1), toPtr(0), a, toPtr(1), x, af, toPtr(2), w, &info)
	Chkxer("DORM2R", &info, lerr, ok, t)
	*infot = 7
	golapack.Dorm2r('R', 'N', toPtr(1), toPtr(2), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, &info)
	Chkxer("DORM2R", &info, lerr, ok, t)
	*infot = 10
	golapack.Dorm2r('L', 'N', toPtr(2), toPtr(1), toPtr(0), a, toPtr(2), x, af, toPtr(1), w, &info)
	Chkxer("DORM2R", &info, lerr, ok, t)

	//     Print a summary line.
	Alaesm(path, ok)
}
