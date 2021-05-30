package lin

import (
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"testing"
)

// Derrrq tests the error exits for the DOUBLE PRECISION routines
// that use the RQ decomposition of a general matrix.
func Derrrq(path []byte, t *testing.T) {
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

	//     Error exits for RQ factorization
	//     DGERQF
	*srnamt = "DGERQF"
	*infot = 1
	golapack.Dgerqf(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGERQF", &info, lerr, ok, t)
	*infot = 2
	golapack.Dgerqf(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGERQF", &info, lerr, ok, t)
	*infot = 4
	golapack.Dgerqf(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, w, func() *int { y := 2; return &y }(), &info)
	Chkxer("DGERQF", &info, lerr, ok, t)
	*infot = 7
	golapack.Dgerqf(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 2; return &y }(), b, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DGERQF", &info, lerr, ok, t)

	//     DGERQ2
	*srnamt = "DGERQ2"
	*infot = 1
	golapack.Dgerq2(toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), b, w, &info)
	Chkxer("DGERQ2", &info, lerr, ok, t)
	*infot = 2
	golapack.Dgerq2(func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), b, w, &info)
	Chkxer("DGERQ2", &info, lerr, ok, t)
	*infot = 4
	golapack.Dgerq2(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), a, func() *int { y := 1; return &y }(), b, w, &info)
	Chkxer("DGERQ2", &info, lerr, ok, t)

	//     DGERQS
	*srnamt = "DGERQS"
	*infot = 1
	Dgerqs(toPtr(-1), toPtr(0), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DGERQS", &info, lerr, ok, t)
	*infot = 2
	Dgerqs(toPtr(0), toPtr(-1), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DGERQS", &info, lerr, ok, t)
	*infot = 2
	Dgerqs(toPtr(2), toPtr(1), toPtr(0), a, toPtr(2), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DGERQS", &info, lerr, ok, t)
	*infot = 3
	Dgerqs(toPtr(0), toPtr(0), toPtr(-1), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DGERQS", &info, lerr, ok, t)
	*infot = 5
	Dgerqs(toPtr(2), toPtr(2), toPtr(0), a, toPtr(1), x, af, toPtr(2), w, toPtr(1), &info)
	Chkxer("DGERQS", &info, lerr, ok, t)
	*infot = 8
	Dgerqs(toPtr(2), toPtr(2), toPtr(0), a, toPtr(2), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DGERQS", &info, lerr, ok, t)
	*infot = 10
	Dgerqs(toPtr(1), toPtr(1), toPtr(2), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DGERQS", &info, lerr, ok, t)

	//     DORGRQ
	*srnamt = "DORGRQ"
	*infot = 1
	golapack.Dorgrq(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DORGRQ", &info, lerr, ok, t)
	*infot = 2
	golapack.Dorgrq(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DORGRQ", &info, lerr, ok, t)
	*infot = 2
	golapack.Dorgrq(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, w, func() *int { y := 2; return &y }(), &info)
	Chkxer("DORGRQ", &info, lerr, ok, t)
	*infot = 3
	golapack.Dorgrq(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DORGRQ", &info, lerr, ok, t)
	*infot = 3
	golapack.Dorgrq(func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DORGRQ", &info, lerr, ok, t)
	*infot = 5
	golapack.Dorgrq(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, func() *int { y := 2; return &y }(), &info)
	Chkxer("DORGRQ", &info, lerr, ok, t)
	*infot = 8
	golapack.Dorgrq(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, w, func() *int { y := 1; return &y }(), &info)
	Chkxer("DORGRQ", &info, lerr, ok, t)

	//     DORGR2
	*srnamt = "DORGR2"
	*infot = 1
	golapack.Dorgr2(toPtr(-1), func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("DORGR2", &info, lerr, ok, t)
	*infot = 2
	golapack.Dorgr2(func() *int { y := 0; return &y }(), toPtr(-1), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("DORGR2", &info, lerr, ok, t)
	*infot = 2
	golapack.Dorgr2(func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 2; return &y }(), x, w, &info)
	Chkxer("DORGR2", &info, lerr, ok, t)
	*infot = 3
	golapack.Dorgr2(func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtr(-1), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("DORGR2", &info, lerr, ok, t)
	*infot = 3
	golapack.Dorgr2(func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a, func() *int { y := 2; return &y }(), x, w, &info)
	Chkxer("DORGR2", &info, lerr, ok, t)
	*infot = 5
	golapack.Dorgr2(func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), func() *int { y := 0; return &y }(), a, func() *int { y := 1; return &y }(), x, w, &info)
	Chkxer("DORGR2", &info, lerr, ok, t)

	//     DORMRQ
	*srnamt = "DORMRQ"
	*infot = 1
	golapack.Dormrq('/', 'N', toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DORMRQ", &info, lerr, ok, t)
	*infot = 2
	golapack.Dormrq('L', '/', toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DORMRQ", &info, lerr, ok, t)
	*infot = 3
	golapack.Dormrq('L', 'N', toPtr(-1), toPtr(0), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DORMRQ", &info, lerr, ok, t)
	*infot = 4
	golapack.Dormrq('L', 'N', toPtr(0), toPtr(-1), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DORMRQ", &info, lerr, ok, t)
	*infot = 5
	golapack.Dormrq('L', 'N', toPtr(0), toPtr(0), toPtr(-1), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DORMRQ", &info, lerr, ok, t)
	*infot = 5
	golapack.Dormrq('L', 'N', toPtr(0), toPtr(1), toPtr(1), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DORMRQ", &info, lerr, ok, t)
	*infot = 5
	golapack.Dormrq('R', 'N', toPtr(1), toPtr(0), toPtr(1), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DORMRQ", &info, lerr, ok, t)
	*infot = 7
	golapack.Dormrq('L', 'N', toPtr(2), toPtr(1), toPtr(2), a, toPtr(1), x, af, toPtr(2), w, toPtr(1), &info)
	Chkxer("DORMRQ", &info, lerr, ok, t)
	*infot = 7
	golapack.Dormrq('R', 'N', toPtr(1), toPtr(2), toPtr(2), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DORMRQ", &info, lerr, ok, t)
	*infot = 10
	golapack.Dormrq('L', 'N', toPtr(2), toPtr(1), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DORMRQ", &info, lerr, ok, t)
	*infot = 12
	golapack.Dormrq('L', 'N', toPtr(1), toPtr(2), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, toPtr(1), &info)
	Chkxer("DORMRQ", &info, lerr, ok, t)
	*infot = 12
	golapack.Dormrq('R', 'N', toPtr(2), toPtr(1), toPtr(0), a, toPtr(1), x, af, toPtr(2), w, toPtr(1), &info)
	Chkxer("DORMRQ", &info, lerr, ok, t)

	//     DORMR2
	*srnamt = "DORMR2"
	*infot = 1
	golapack.Dormr2('/', 'N', toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, &info)
	Chkxer("DORMR2", &info, lerr, ok, t)
	*infot = 2
	golapack.Dormr2('L', '/', toPtr(0), toPtr(0), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, &info)
	Chkxer("DORMR2", &info, lerr, ok, t)
	*infot = 3
	golapack.Dormr2('L', 'N', toPtr(-1), toPtr(0), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, &info)
	Chkxer("DORMR2", &info, lerr, ok, t)
	*infot = 4
	golapack.Dormr2('L', 'N', toPtr(0), toPtr(-1), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, &info)
	Chkxer("DORMR2", &info, lerr, ok, t)
	*infot = 5
	golapack.Dormr2('L', 'N', toPtr(0), toPtr(0), toPtr(-1), a, toPtr(1), x, af, toPtr(1), w, &info)
	Chkxer("DORMR2", &info, lerr, ok, t)
	*infot = 5
	golapack.Dormr2('L', 'N', toPtr(0), toPtr(1), toPtr(1), a, toPtr(1), x, af, toPtr(1), w, &info)
	Chkxer("DORMR2", &info, lerr, ok, t)
	*infot = 5
	golapack.Dormr2('R', 'N', toPtr(1), toPtr(0), toPtr(1), a, toPtr(1), x, af, toPtr(1), w, &info)
	Chkxer("DORMR2", &info, lerr, ok, t)
	*infot = 7
	golapack.Dormr2('L', 'N', toPtr(2), toPtr(1), toPtr(2), a, toPtr(1), x, af, toPtr(2), w, &info)
	Chkxer("DORMR2", &info, lerr, ok, t)
	*infot = 7
	golapack.Dormr2('R', 'N', toPtr(1), toPtr(2), toPtr(2), a, toPtr(1), x, af, toPtr(1), w, &info)
	Chkxer("DORMR2", &info, lerr, ok, t)
	*infot = 10
	golapack.Dormr2('L', 'N', toPtr(2), toPtr(1), toPtr(0), a, toPtr(1), x, af, toPtr(1), w, &info)
	Chkxer("DORMR2", &info, lerr, ok, t)

	//     Print a summary line.
	Alaesm(path, ok)
}
