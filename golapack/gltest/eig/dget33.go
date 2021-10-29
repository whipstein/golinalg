package eig

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
)

// dget33 tests DLANV2, a routine for putting 2 by 2 blocks into
// standard form.  In other words, it computes a two by two rotation
// [[C,S];[-S,C]] where in
//
//    [ C S ][T(1,1) T(1,2)][ C -S ] = [ T11 T12 ]
//    [-S C ][T(2,1) T(2,2)][ S  C ]   [ T21 T22 ]
//
// either
//    1) T21=0 (real eigenvalues), or
//    2) T11=T22 and T21*T12<0 (complex conjugate eigenvalues).
// We also  verify that the residual is small.
func dget33() (rmax float64, lmax, ninfo, knt int) {
	var bignum, cs, eps, four, one, res, smlnum, sn, sum, tnrm, two, zero float64
	var i1, i2, i3, i4, im1, im2, im3, im4, j1, j2, j3 int

	val := vf(4)
	vm := vf(3)
	q := mf(2, 2, opts)
	t := mf(2, 2, opts)
	t1 := mf(2, 2, opts)
	t2 := mf(2, 2, opts)

	zero = 0.0
	one = 1.0
	two = 2.0
	four = 4.0

	//     Get machine parameters
	eps = golapack.Dlamch(Precision)
	smlnum = golapack.Dlamch(SafeMinimum) / eps
	bignum = one / smlnum
	smlnum, bignum = golapack.Dlabad(smlnum, bignum)

	//     Set up test case parameters
	val.Set(0, one)
	val.Set(1, one+two*eps)
	val.Set(2, two)
	val.Set(3, two-four*eps)
	vm.Set(0, smlnum)
	vm.Set(1, one)
	vm.Set(2, bignum)

	knt = 0
	ninfo = 0
	lmax = 0
	rmax = zero

	//     Begin test loop
	for i1 = 1; i1 <= 4; i1++ {
		for i2 = 1; i2 <= 4; i2++ {
			for i3 = 1; i3 <= 4; i3++ {
				for i4 = 1; i4 <= 4; i4++ {
					for im1 = 1; im1 <= 3; im1++ {
						for im2 = 1; im2 <= 3; im2++ {
							for im3 = 1; im3 <= 3; im3++ {
								for im4 = 1; im4 <= 3; im4++ {
									t.Set(0, 0, val.Get(i1-1)*vm.Get(im1-1))
									t.Set(0, 1, val.Get(i2-1)*vm.Get(im2-1))
									t.Set(1, 0, -val.Get(i3-1)*vm.Get(im3-1))
									t.Set(1, 1, val.Get(i4-1)*vm.Get(im4-1))
									tnrm = math.Max(math.Abs(t.Get(0, 0)), math.Max(math.Abs(t.Get(0, 1)), math.Max(math.Abs(t.Get(1, 0)), math.Abs(t.Get(1, 1)))))
									t1.Set(0, 0, t.Get(0, 0))
									t1.Set(0, 1, t.Get(0, 1))
									t1.Set(1, 0, t.Get(1, 0))
									t1.Set(1, 1, t.Get(1, 1))
									q.Set(0, 0, one)
									q.Set(0, 1, zero)
									q.Set(1, 0, zero)
									q.Set(1, 1, one)

									*t.GetPtr(0, 0), *t.GetPtr(0, 1), *t.GetPtr(1, 0), *t.GetPtr(1, 1), _, _, _, _, cs, sn = golapack.Dlanv2(t.Get(0, 0), t.Get(0, 1), t.Get(1, 0), t.Get(1, 1))
									for j1 = 1; j1 <= 2; j1++ {
										res = q.Get(j1-1, 0)*cs + q.Get(j1-1, 1)*sn
										q.Set(j1-1, 1, -q.Get(j1-1, 0)*sn+q.Get(j1-1, 1)*cs)
										q.Set(j1-1, 0, res)
									}

									res = zero
									res = res + math.Abs(math.Pow(q.Get(0, 0), 2)+math.Pow(q.Get(0, 1), 2)-one)/eps
									res = res + math.Abs(math.Pow(q.Get(1, 1), 2)+math.Pow(q.Get(1, 0), 2)-one)/eps
									res = res + math.Abs(q.Get(0, 0)*q.Get(1, 0)+q.Get(0, 1)*q.Get(1, 1))/eps
									for j1 = 1; j1 <= 2; j1++ {
										for j2 = 1; j2 <= 2; j2++ {
											t2.Set(j1-1, j2-1, zero)
											for j3 = 1; j3 <= 2; j3++ {
												t2.Set(j1-1, j2-1, t2.Get(j1-1, j2-1)+t1.Get(j1-1, j3-1)*q.Get(j3-1, j2-1))
											}
										}
									}
									for j1 = 1; j1 <= 2; j1++ {
										for j2 = 1; j2 <= 2; j2++ {
											sum = t.Get(j1-1, j2-1)
											for j3 = 1; j3 <= 2; j3++ {
												sum = sum - q.Get(j3-1, j1-1)*t2.Get(j3-1, j2-1)
											}
											res = res + math.Abs(sum)/eps/tnrm
										}
									}
									if t.Get(1, 0) != zero && (t.Get(0, 0) != t.Get(1, 1) || math.Copysign(one, t.Get(0, 1))*math.Copysign(one, t.Get(1, 0)) > zero) {
										res = res + one/eps
									}
									knt = knt + 1
									if res > rmax {
										lmax = knt
										rmax = res
									}
								}
							}
						}
					}
				}
			}
		}
	}

	return
}
