package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlalsd uses the singular value decomposition of A to solve the least
// squares problem of finding X to minimize the Euclidean norm of each
// column of A*X-B, where A is N-by-N upper bidiagonal, and X and B
// are N-by-NRHS. The solution X overwrites B.
//
// The singular values of A smaller than RCOND times the largest
// singular value are treated as zero in solving the least squares
// problem; in this case a minimum norm solution is returned.
// The actual singular values are returned in D in ascending order.
//
// This code makes very mild assumptions about floating point
// arithmetic. It will work on machines with a guard digit in
// add/subtract, or on those binary machines without guard digits
// which subtract like the Cray XMP, Cray YMP, Cray C 90, or Cray 2.
// It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.
func Dlalsd(uplo byte, smlsiz, n, nrhs *int, d, e *mat.Vector, b *mat.Matrix, ldb *int, rcond *float64, rank *int, work *mat.Vector, iwork *[]int, info *int) {
	var cs, eps, one, orgnrm, r, rcnd, sn, tol, two, zero float64
	var bx, bxst, c, difl, difr, givcol, givnum, givptr, i, icmpq1, icmpq2, iwk, j, k, nlvl, nm1, nsize, nsub, nwork, perm, poles, s, sizei, smlszp, sqre, st, st1, u, vt, z int
	var err error
	_ = err

	zero = 0.0
	one = 1.0
	two = 2.0

	//     Test the input parameters.
	(*info) = 0

	if (*n) < 0 {
		(*info) = -3
	} else if (*nrhs) < 1 {
		(*info) = -4
	} else if ((*ldb) < 1) || ((*ldb) < (*n)) {
		(*info) = -8
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLALSD"), -(*info))
		return
	}

	eps = Dlamch(Epsilon)

	//     Set up the tolerance.
	if ((*rcond) <= zero) || ((*rcond) >= one) {
		rcnd = eps
	} else {
		rcnd = (*rcond)
	}

	(*rank) = 0

	//     Quick return if possible.
	if (*n) == 0 {
		return
	} else if (*n) == 1 {
		if d.Get(0) == zero {
			Dlaset('A', func() *int { y := 1; return &y }(), nrhs, &zero, &zero, b, ldb)
		} else {
			(*rank) = 1
			Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), d.GetPtr(0), &one, func() *int { y := 1; return &y }(), nrhs, b, ldb, info)
			d.Set(0, math.Abs(d.Get(0)))
		}
		return
	}

	//     Rotate the matrix if it is lower bidiagonal.
	if uplo == 'L' {
		for i = 1; i <= (*n)-1; i++ {
			Dlartg(d.GetPtr(i-1), e.GetPtr(i-1), &cs, &sn, &r)
			d.Set(i-1, r)
			e.Set(i-1, sn*d.Get(i))
			d.Set(i, cs*d.Get(i))
			if (*nrhs) == 1 {
				goblas.Drot(1, b.Vector(i-1, 0, 1), b.Vector(i, 0, 1), cs, sn)
			} else {
				work.Set(i*2-1-1, cs)
				work.Set(i*2-1, sn)
			}
		}
		if (*nrhs) > 1 {
			for i = 1; i <= (*nrhs); i++ {
				for j = 1; j <= (*n)-1; j++ {
					cs = work.Get(j*2 - 1 - 1)
					sn = work.Get(j*2 - 1)
					goblas.Drot(1, b.Vector(j-1, i-1, 1), b.Vector(j, i-1, 1), cs, sn)
				}
			}
		}
	}

	//     Scale.
	nm1 = (*n) - 1
	orgnrm = Dlanst('M', n, d, e)
	if orgnrm == zero {
		Dlaset('A', n, nrhs, &zero, &zero, b, ldb)
		return
	}

	Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &orgnrm, &one, n, func() *int { y := 1; return &y }(), d.Matrix(*n, opts), n, info)
	Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &orgnrm, &one, &nm1, func() *int { y := 1; return &y }(), e.Matrix(nm1, opts), &nm1, info)

	//     If N is smaller than the minimum divide size SMLSIZ, then solve
	//     the problem with another solver.
	if (*n) <= (*smlsiz) {
		nwork = 1 + (*n)*(*n)
		Dlaset('A', n, n, &zero, &one, work.Matrix(*n, opts), n)
		Dlasdq('U', func() *int { y := 0; return &y }(), n, n, func() *int { y := 0; return &y }(), nrhs, d, e, work.Matrix(*n, opts), n, work.Matrix(*n, opts), n, b, ldb, work.Off(nwork-1), info)
		if (*info) != 0 {
			return
		}
		tol = rcnd * math.Abs(d.Get(goblas.Idamax(*n, d)-1))
		for i = 1; i <= (*n); i++ {
			if d.Get(i-1) <= tol {
				Dlaset('A', func() *int { y := 1; return &y }(), nrhs, &zero, &zero, b.Off(i-1, 0), ldb)
			} else {
				Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), d.GetPtr(i-1), &one, func() *int { y := 1; return &y }(), nrhs, b.Off(i-1, 0), ldb, info)
				(*rank) = (*rank) + 1
			}
		}
		err = goblas.Dgemm(Trans, NoTrans, *n, *nrhs, *n, one, work.Matrix(*n, opts), b, zero, work.MatrixOff(nwork-1, *n, opts))
		Dlacpy('A', n, nrhs, work.MatrixOff(nwork-1, *n, opts), n, b, ldb)

		//        Unscale.
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &one, &orgnrm, n, func() *int { y := 1; return &y }(), d.Matrix(*n, opts), n, info)
		Dlasrt('D', n, d, info)
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &orgnrm, &one, n, nrhs, b, ldb, info)

		return
	}

	//     Book-keeping and setting up some constants.
	nlvl = int(math.Log(float64(*n)/float64((*smlsiz)+1))/math.Log(two)) + 1

	smlszp = (*smlsiz) + 1

	u = 1
	vt = 1 + (*smlsiz)*(*n)
	difl = vt + smlszp*(*n)
	difr = difl + nlvl*(*n)
	z = difr + nlvl*(*n)*2
	c = z + nlvl*(*n)
	s = c + (*n)
	poles = s + (*n)
	givnum = poles + 2*nlvl*(*n)
	bx = givnum + 2*nlvl*(*n)
	nwork = bx + (*n)*(*nrhs)

	sizei = 1 + (*n)
	k = sizei + (*n)
	givptr = k + (*n)
	perm = givptr + (*n)
	givcol = perm + nlvl*(*n)
	iwk = givcol + nlvl*(*n)*2

	st = 1
	sqre = 0
	icmpq1 = 1
	icmpq2 = 0
	nsub = 0

	for i = 1; i <= (*n); i++ {
		if math.Abs(d.Get(i-1)) < eps {
			d.Set(i-1, math.Copysign(eps, d.Get(i-1)))
		}
	}

	for i = 1; i <= nm1; i++ {
		if (math.Abs(e.Get(i-1)) < eps) || (i == nm1) {
			nsub = nsub + 1
			(*iwork)[nsub-1] = st

			//           Subproblem found. First determine its size and then
			//           apply divide and conquer on it.
			if i < nm1 {
				//              A subproblem with E(I) small for I < NM1.
				nsize = i - st + 1
				(*iwork)[sizei+nsub-1-1] = nsize
			} else if math.Abs(e.Get(i-1)) >= eps {
				//              A subproblem with E(NM1) not too small but I = NM1.
				nsize = (*n) - st + 1
				(*iwork)[sizei+nsub-1-1] = nsize
			} else {
				//              A subproblem with E(NM1) small. This implies an
				//              1-by-1 subproblem at D(N), which is not solved
				//              explicitly.
				nsize = i - st + 1
				(*iwork)[sizei+nsub-1-1] = nsize
				nsub = nsub + 1
				(*iwork)[nsub-1] = (*n)
				(*iwork)[sizei+nsub-1-1] = 1
				goblas.Dcopy(*nrhs, b.Vector((*n)-1, 0), work.Off(bx+nm1-1, *n))
			}
			st1 = st - 1
			if nsize == 1 {
				//              This is a 1-by-1 subproblem and is not solved
				//              explicitly.
				goblas.Dcopy(*nrhs, b.Vector(st-1, 0), work.Off(bx+st1-1, *n))
			} else if nsize <= (*smlsiz) {
				//              This is a small subproblem and is solved by DLASDQ.
				Dlaset('A', &nsize, &nsize, &zero, &one, work.MatrixOff(vt+st1-1, *n, opts), n)
				Dlasdq('U', func() *int { y := 0; return &y }(), &nsize, &nsize, func() *int { y := 0; return &y }(), nrhs, d.Off(st-1), e.Off(st-1), work.MatrixOff(vt+st1-1, *n, opts), n, work.MatrixOff(nwork-1, *n, opts), n, b.Off(st-1, 0), ldb, work.Off(nwork-1), info)
				if (*info) != 0 {
					return
				}
				Dlacpy('A', &nsize, nrhs, b.Off(st-1, 0), ldb, work.MatrixOff(bx+st1-1, *n, opts), n)
			} else {
				//              A large problem. Solve it using divide and conquer.
				Dlasda(&icmpq1, smlsiz, &nsize, &sqre, d.Off(st-1), e.Off(st-1), work.MatrixOff(u+st1-1, *n, opts), n, work.MatrixOff(vt+st1-1, *n, opts), toSlice(iwork, k+st1-1), work.MatrixOff(difl+st1-1, *n, opts), work.MatrixOff(difr+st1-1, *n, opts), work.MatrixOff(z+st1-1, *n, opts), work.MatrixOff(poles+st1-1, *n, opts), toSlice(iwork, givptr+st1-1), toSlice(iwork, givcol+st1-1), n, toSlice(iwork, perm+st1-1), work.MatrixOff(givnum+st1-1, *n, opts), work.Off(c+st1-1), work.Off(s+st1-1), work.Off(nwork-1), toSlice(iwork, iwk-1), info)
				if (*info) != 0 {
					return
				}
				bxst = bx + st1
				Dlalsa(&icmpq2, smlsiz, &nsize, nrhs, b.Off(st-1, 0), ldb, work.MatrixOff(bxst-1, *n, opts), n, work.MatrixOff(u+st1-1, *n, opts), n, work.MatrixOff(vt+st1-1, *n, opts), toSlice(iwork, k+st1-1), work.MatrixOff(difl+st1-1, *n, opts), work.MatrixOff(difr+st1-1, *n, opts), work.MatrixOff(z+st1-1, *n, opts), work.MatrixOff(poles+st1-1, *n, opts), toSlice(iwork, givptr+st1-1), toSlice(iwork, givcol+st1-1), n, toSlice(iwork, perm+st1-1), work.MatrixOff(givnum+st1-1, *n, opts), work.Off(c+st1-1), work.Off(s+st1-1), work.Off(nwork-1), toSlice(iwork, iwk-1), info)
				if (*info) != 0 {
					return
				}
			}
			st = i + 1
		}
	}

	//     Apply the singular values and treat the tiny ones as zero.
	tol = rcnd * math.Abs(d.Get(goblas.Idamax(*n, d)-1))

	for i = 1; i <= (*n); i++ {
		//        Some of the elements in D can be negative because 1-by-1
		//        subproblems were not solved explicitly.
		if math.Abs(d.Get(i-1)) <= tol {
			Dlaset('A', func() *int { y := 1; return &y }(), nrhs, &zero, &zero, work.MatrixOff(bx+i-1-1, *n, opts), n)
		} else {
			(*rank) = (*rank) + 1
			Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), d.GetPtr(i-1), &one, func() *int { y := 1; return &y }(), nrhs, work.MatrixOff(bx+i-1-1, *n, opts), n, info)
		}
		d.Set(i-1, math.Abs(d.Get(i-1)))
	}

	//     Now apply back the right singular vectors.
	icmpq2 = 1
	for i = 1; i <= nsub; i++ {
		st = (*iwork)[i-1]
		st1 = st - 1
		nsize = (*iwork)[sizei+i-1-1]
		bxst = bx + st1
		if nsize == 1 {
			goblas.Dcopy(*nrhs, work.Off(bxst-1, *n), b.Vector(st-1, 0))
		} else if nsize <= (*smlsiz) {
			err = goblas.Dgemm(Trans, NoTrans, nsize, *nrhs, nsize, one, work.MatrixOff(vt+st1-1, *n, opts), work.MatrixOff(bxst-1, *n, opts), zero, b.Off(st-1, 0))
		} else {
			Dlalsa(&icmpq2, smlsiz, &nsize, nrhs, work.MatrixOff(bxst-1, *n, opts), n, b.Off(st-1, 0), ldb, work.MatrixOff(u+st1-1, *n, opts), n, work.MatrixOff(vt+st1-1, *n, opts), toSlice(iwork, k+st1-1), work.MatrixOff(difl+st1-1, *n, opts), work.MatrixOff(difr+st1-1, *n, opts), work.MatrixOff(z+st1-1, *n, opts), work.MatrixOff(poles+st1-1, *n, opts), toSlice(iwork, givptr+st1-1), toSlice(iwork, givcol+st1-1), n, toSlice(iwork, perm+st1-1), work.MatrixOff(givnum+st1-1, *n, opts), work.Off(c+st1-1), work.Off(s+st1-1), work.Off(nwork-1), toSlice(iwork, iwk-1), info)
			if (*info) != 0 {
				return
			}
		}
	}

	//     Unscale and sort the singular values.
	Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &one, &orgnrm, n, func() *int { y := 1; return &y }(), d.Matrix(*n, opts), n, info)
	Dlasrt('D', n, d, info)
	Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &orgnrm, &one, n, nrhs, b, ldb, info)
}
