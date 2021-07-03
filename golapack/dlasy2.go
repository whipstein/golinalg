package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Dlasy2 solves for the N1 by N2 matrix X, 1 <= N1,N2 <= 2, in
//
//        op(TL)*X + ISGN*X*op(TR) = SCALE*B,
//
// where TL is N1 by N1, TR is N2 by N2, B is N1 by N2, and ISGN = 1 or
// -1.  op(T) = T or T**T, where T**T denotes the transpose of T.
func Dlasy2(ltranl, ltranr bool, isgn, n1, n2 *int, tl *mat.Matrix, ldtl *int, tr *mat.Matrix, ldtr *int, b *mat.Matrix, ldb *int, scale *float64, x *mat.Matrix, ldx *int, xnorm *float64, info *int) {
	var bswap, xswap bool
	var bet, eight, eps, gam, half, l21, one, sgn, smin, smlnum, tau1, temp, two, u11, u12, u22, xmax, zero float64
	var i, ip, ipiv, ipsv, j, jp, jpsv, k int
	bswpiv := []bool{false, true, false, true}
	xswpiv := []bool{false, false, true, true}
	jpiv := make([]int, 4)
	locl21 := []int{2, 1, 4, 3}
	locu12 := []int{3, 4, 1, 2}
	locu22 := []int{4, 3, 2, 1}

	btmp := vf(4)
	tmp := vf(4)
	x2 := vf(2)
	t16 := mf(4, 4, opts)

	zero = 0.0
	one = 1.0
	two = 2.0
	half = 0.5
	eight = 8.0

	//     Do not check the input parameters for errors
	(*info) = 0

	//     Quick return if possible
	if (*n1) == 0 || (*n2) == 0 {
		return
	}

	//     Set constants to control overflow
	eps = Dlamch(Precision)
	smlnum = Dlamch(SafeMinimum) / eps
	sgn = float64(*isgn)

	k = (*n1) + (*n1) + (*n2) - 2
	switch k {
	case 1:
		goto label10
	case 2:
		goto label20
	case 3:
		goto label30
	case 4:
		goto label50
	}

	//     1 by 1: TL11*X + SGN*X*TR11 = B11
label10:
	;
	tau1 = tl.Get(0, 0) + sgn*tr.Get(0, 0)
	bet = math.Abs(tau1)
	if bet <= smlnum {
		tau1 = smlnum
		bet = smlnum
		(*info) = 1
	}

	(*scale) = one
	gam = math.Abs(b.Get(0, 0))
	if smlnum*gam > bet {
		(*scale) = one / gam
	}

	x.Set(0, 0, (b.Get(0, 0)*(*scale))/tau1)
	(*xnorm) = math.Abs(x.Get(0, 0))
	return

	//     1 by 2:
	//     TL11*[X11 X12] + ISGN*[X11 X12]*op[TR11 TR12]  = [B11 B12]
	//                                       [TR21 TR22]
label20:
	;

	smin = maxf64(eps*maxf64(math.Abs(tl.Get(0, 0)), math.Abs(tr.Get(0, 0)), math.Abs(tr.Get(0, 1)), math.Abs(tr.Get(1, 0)), math.Abs(tr.Get(1, 1))), smlnum)
	tmp.Set(0, tl.Get(0, 0)+sgn*tr.Get(0, 0))
	tmp.Set(3, tl.Get(0, 0)+sgn*tr.Get(1, 1))
	if ltranr {
		tmp.Set(1, sgn*tr.Get(1, 0))
		tmp.Set(2, sgn*tr.Get(0, 1))
	} else {
		tmp.Set(1, sgn*tr.Get(0, 1))
		tmp.Set(2, sgn*tr.Get(1, 0))
	}
	btmp.Set(0, b.Get(0, 0))
	btmp.Set(1, b.Get(0, 1))
	goto label40

	//     2 by 1:
	//          op[TL11 TL12]*[X11] + ISGN* [X11]*TR11  = [B11]
	//            [TL21 TL22] [X21]         [X21]         [B21]
label30:
	;
	smin = maxf64(eps*maxf64(math.Abs(tr.Get(0, 0)), math.Abs(tl.Get(0, 0)), math.Abs(tl.Get(0, 1)), math.Abs(tl.Get(1, 0)), math.Abs(tl.Get(1, 1))), smlnum)
	tmp.Set(0, tl.Get(0, 0)+sgn*tr.Get(0, 0))
	tmp.Set(3, tl.Get(1, 1)+sgn*tr.Get(0, 0))
	if ltranl {
		tmp.Set(1, tl.Get(0, 1))
		tmp.Set(2, tl.Get(1, 0))
	} else {
		tmp.Set(1, tl.Get(1, 0))
		tmp.Set(2, tl.Get(0, 1))
	}
	btmp.Set(0, b.Get(0, 0))
	btmp.Set(1, b.Get(1, 0))
label40:
	;

	//     Solve 2 by 2 system using complete pivoting.
	//     Set pivots less than SMIN to SMIN.
	ipiv = goblas.Idamax(4, tmp, 1)
	u11 = tmp.Get(ipiv - 1)
	if math.Abs(u11) <= smin {
		(*info) = 1
		u11 = smin
	}
	u12 = tmp.Get(locu12[ipiv-1] - 1)
	l21 = tmp.Get(locl21[ipiv-1]-1) / u11
	u22 = tmp.Get(locu22[ipiv-1]-1) - u12*l21
	xswap = xswpiv[ipiv-1]
	bswap = bswpiv[ipiv-1]
	if math.Abs(u22) <= smin {
		(*info) = 1
		u22 = smin
	}
	if bswap {
		temp = btmp.Get(1)
		btmp.Set(1, btmp.Get(0)-l21*temp)
		btmp.Set(0, temp)
	} else {
		btmp.Set(1, btmp.Get(1)-l21*btmp.Get(0))
	}
	(*scale) = one
	if (two*smlnum)*math.Abs(btmp.Get(1)) > math.Abs(u22) || (two*smlnum)*math.Abs(btmp.Get(0)) > math.Abs(u11) {
		(*scale) = half / maxf64(math.Abs(btmp.Get(0)), math.Abs(btmp.Get(1)))
		btmp.Set(0, btmp.Get(0)*(*scale))
		btmp.Set(1, btmp.Get(1)*(*scale))
	}
	x2.Set(1, btmp.Get(1)/u22)
	x2.Set(0, btmp.Get(0)/u11-(u12/u11)*x2.Get(1))
	if xswap {
		temp = x2.Get(1)
		x2.Set(1, x2.Get(0))
		x2.Set(0, temp)
	}
	x.Set(0, 0, x2.Get(0))
	if (*n1) == 1 {
		x.Set(0, 1, x2.Get(1))
		(*xnorm) = math.Abs(x.Get(0, 0)) + math.Abs(x.Get(0, 1))
	} else {
		x.Set(1, 0, x2.Get(1))
		(*xnorm) = maxf64(math.Abs(x.Get(0, 0)), math.Abs(x.Get(1, 0)))
	}
	return

	//     2 by 2:
	//     op[TL11 TL12]*[X11 X12] +ISGN* [X11 X12]*op[TR11 TR12] = [B11 B12]
	//       [TL21 TL22] [X21 X22]        [X21 X22]   [TR21 TR22]   [B21 B22]
	//
	//     Solve equivalent 4 by 4 system using complete pivoting.
	//     Set pivots less than SMIN to SMIN.
label50:
	;
	smin = maxf64(math.Abs(tr.Get(0, 0)), math.Abs(tr.Get(0, 1)), math.Abs(tr.Get(1, 0)), math.Abs(tr.Get(1, 1)))
	smin = maxf64(smin, math.Abs(tl.Get(0, 0)), math.Abs(tl.Get(0, 1)), math.Abs(tl.Get(1, 0)), math.Abs(tl.Get(1, 1)))
	smin = maxf64(eps*smin, smlnum)
	btmp.Set(0, zero)
	goblas.Dcopy(16, btmp, 0, t16.VectorIdx(0), 1)
	t16.Set(0, 0, tl.Get(0, 0)+sgn*tr.Get(0, 0))
	t16.Set(1, 1, tl.Get(1, 1)+sgn*tr.Get(0, 0))
	t16.Set(2, 2, tl.Get(0, 0)+sgn*tr.Get(1, 1))
	t16.Set(3, 3, tl.Get(1, 1)+sgn*tr.Get(1, 1))
	if ltranl {
		t16.Set(0, 1, tl.Get(1, 0))
		t16.Set(1, 0, tl.Get(0, 1))
		t16.Set(2, 3, tl.Get(1, 0))
		t16.Set(3, 2, tl.Get(0, 1))
	} else {
		t16.Set(0, 1, tl.Get(0, 1))
		t16.Set(1, 0, tl.Get(1, 0))
		t16.Set(2, 3, tl.Get(0, 1))
		t16.Set(3, 2, tl.Get(1, 0))
	}
	if ltranr {
		t16.Set(0, 2, sgn*tr.Get(0, 1))
		t16.Set(1, 3, sgn*tr.Get(0, 1))
		t16.Set(2, 0, sgn*tr.Get(1, 0))
		t16.Set(3, 1, sgn*tr.Get(1, 0))
	} else {
		t16.Set(0, 2, sgn*tr.Get(1, 0))
		t16.Set(1, 3, sgn*tr.Get(1, 0))
		t16.Set(2, 0, sgn*tr.Get(0, 1))
		t16.Set(3, 1, sgn*tr.Get(0, 1))
	}
	btmp.Set(0, b.Get(0, 0))
	btmp.Set(1, b.Get(1, 0))
	btmp.Set(2, b.Get(0, 1))
	btmp.Set(3, b.Get(1, 1))

	//     Perform elimination
	for i = 1; i <= 3; i++ {
		xmax = zero
		for ip = i; ip <= 4; ip++ {
			for jp = i; jp <= 4; jp++ {
				if math.Abs(t16.Get(ip-1, jp-1)) >= xmax {
					xmax = math.Abs(t16.Get(ip-1, jp-1))
					ipsv = ip
					jpsv = jp
				}
			}
		}
		if ipsv != i {
			goblas.Dswap(4, t16.Vector(ipsv-1, 0), 4, t16.Vector(i-1, 0), 4)
			temp = btmp.Get(i - 1)
			btmp.Set(i-1, btmp.Get(ipsv-1))
			btmp.Set(ipsv-1, temp)
		}
		if jpsv != i {
			goblas.Dswap(4, t16.Vector(0, jpsv-1), 1, t16.Vector(0, i-1), 1)
		}
		jpiv[i-1] = jpsv
		if math.Abs(t16.Get(i-1, i-1)) < smin {
			(*info) = 1
			t16.Set(i-1, i-1, smin)
		}
		for j = i + 1; j <= 4; j++ {
			t16.Set(j-1, i-1, t16.Get(j-1, i-1)/t16.Get(i-1, i-1))
			btmp.Set(j-1, btmp.Get(j-1)-t16.Get(j-1, i-1)*btmp.Get(i-1))
			for k = i + 1; k <= 4; k++ {
				t16.Set(j-1, k-1, t16.Get(j-1, k-1)-t16.Get(j-1, i-1)*t16.Get(i-1, k-1))
			}
		}
	}
	if math.Abs(t16.Get(3, 3)) < smin {
		(*info) = 1
		t16.Set(3, 3, smin)
	}
	(*scale) = one
	if (eight*smlnum)*math.Abs(btmp.Get(0)) > math.Abs(t16.Get(0, 0)) || (eight*smlnum)*math.Abs(btmp.Get(1)) > math.Abs(t16.Get(1, 1)) || (eight*smlnum)*math.Abs(btmp.Get(2)) > math.Abs(t16.Get(2, 2)) || (eight*smlnum)*math.Abs(btmp.Get(3)) > math.Abs(t16.Get(3, 3)) {
		(*scale) = (one / eight) / maxf64(math.Abs(btmp.Get(0)), math.Abs(btmp.Get(1)), math.Abs(btmp.Get(2)), math.Abs(btmp.Get(3)))
		btmp.Set(0, btmp.Get(0)*(*scale))
		btmp.Set(1, btmp.Get(1)*(*scale))
		btmp.Set(2, btmp.Get(2)*(*scale))
		btmp.Set(3, btmp.Get(3)*(*scale))
	}
	for i = 1; i <= 4; i++ {
		k = 5 - i
		temp = one / t16.Get(k-1, k-1)
		tmp.Set(k-1, btmp.Get(k-1)*temp)
		for j = k + 1; j <= 4; j++ {
			tmp.Set(k-1, tmp.Get(k-1)-(temp*t16.Get(k-1, j-1))*tmp.Get(j-1))
		}
	}
	for i = 1; i <= 3; i++ {
		if jpiv[4-i-1] != 4-i {
			temp = tmp.Get(4 - i - 1)
			tmp.Set(4-i-1, tmp.Get(jpiv[4-i-1]-1))
			tmp.Set(jpiv[4-i-1]-1, temp)
		}
	}
	x.Set(0, 0, tmp.Get(0))
	x.Set(1, 0, tmp.Get(1))
	x.Set(0, 1, tmp.Get(2))
	x.Set(1, 1, tmp.Get(3))
	(*xnorm) = maxf64(math.Abs(tmp.Get(0))+math.Abs(tmp.Get(2)), math.Abs(tmp.Get(1))+math.Abs(tmp.Get(3)))
}
