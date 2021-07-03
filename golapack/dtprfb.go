package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Dtprfb applies a real "triangular-pentagonal" block reflector H or its
// transpose H**T to a real matrix C, which is composed of two
// blocks A and B, either from the left or right.
func Dtprfb(side, trans, direct, storev byte, m, n, k, l *int, v *mat.Matrix, ldv *int, t *mat.Matrix, ldt *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, work *mat.Matrix, ldwork *int) {
	var backward, column, forward, left, right, row bool
	var one, zero float64
	var i, j, kp, mp, np int
	var err error
	_ = err

	one = 1.0
	zero = 0.0

	//     Quick return if possible
	if (*m) <= 0 || (*n) <= 0 || (*k) <= 0 || (*l) < 0 {
		return
	}

	if storev == 'C' {
		column = true
		row = false
	} else if storev == 'R' {
		column = false
		row = true
	} else {
		column = false
		row = false
	}

	if side == 'L' {
		left = true
		right = false
	} else if side == 'R' {
		left = false
		right = true
	} else {
		left = false
		right = false
	}

	if direct == 'F' {
		forward = true
		backward = false
	} else if direct == 'B' {
		forward = false
		backward = true
	} else {
		forward = false
		backward = false
	}

	if column && forward && left {
		// ---------------------------------------------------------------------------
		//
		//        Let  W =  [ I ]    (K-by-K)
		//                  [ V ]    (M-by-K)
		//
		//        Form  H C  or  H**T C  where  C = [ A ]  (K-by-N)
		//                                          [ B ]  (M-by-N)
		//
		//        H = I - W T W**T          or  H**T = I - W T**T W**T
		//
		//        A = A -   T (A + V**T B)  or  A = A -   T**T (A + V**T B)
		//        B = B - V T (A + V**T B)  or  B = B - V T**T (A + V**T B)
		//
		// ---------------------------------------------------------------------------
		mp = minint((*m)-(*l)+1, *m)
		kp = minint((*l)+1, *k)

		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*l); i++ {
				work.Set(i-1, j-1, b.Get((*m)-(*l)+i-1, j-1))
			}
		}
		err = goblas.Dtrmm(Left, Upper, Trans, NonUnit, *l, *n, one, v.Off(mp-1, 0), *ldv, work, *ldwork)
		err = goblas.Dgemm(Trans, NoTrans, *l, *n, (*m)-(*l), one, v, *ldv, b, *ldb, one, work, *ldwork)
		err = goblas.Dgemm(Trans, NoTrans, (*k)-(*l), *n, *m, one, v.Off(0, kp-1), *ldv, b, *ldb, zero, work.Off(kp-1, 0), *ldwork)

		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*k); i++ {
				work.Set(i-1, j-1, work.Get(i-1, j-1)+a.Get(i-1, j-1))
			}
		}

		goblas.Dtrmm(Left, Upper, mat.TransByte(trans), NonUnit, *k, *n, one, t, *ldt, work, *ldwork)

		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*k); i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		err = goblas.Dgemm(NoTrans, NoTrans, (*m)-(*l), *n, *k, -one, v, *ldv, work, *ldwork, one, b, *ldb)
		err = goblas.Dgemm(NoTrans, NoTrans, *l, *n, (*k)-(*l), -one, v.Off(mp-1, kp-1), *ldv, work.Off(kp-1, 0), *ldwork, one, b.Off(mp-1, 0), *ldb)
		err = goblas.Dtrmm(Left, Upper, NoTrans, NonUnit, *l, *n, one, v.Off(mp-1, 0), *ldv, work, *ldwork)
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*l); i++ {
				b.Set((*m)-(*l)+i-1, j-1, b.Get((*m)-(*l)+i-1, j-1)-work.Get(i-1, j-1))
			}
		}

	} else if column && forward && right {
		// ---------------------------------------------------------------------------
		//
		//        Let  W =  [ I ]    (K-by-K)
		//                  [ V ]    (N-by-K)
		//
		//        Form  C H or  C H**T  where  C = [ A B ] (A is M-by-K, B is M-by-N)
		//
		//        H = I - W T W**T          or  H**T = I - W T**T W**T
		//
		//        A = A - (A + B V) T      or  A = A - (A + B V) T**T
		//        B = B - (A + B V) T V**T  or  B = B - (A + B V) T**T V**T
		//
		// ---------------------------------------------------------------------------
		np = minint((*n)-(*l)+1, *n)
		kp = minint((*l)+1, *k)

		for j = 1; j <= (*l); j++ {
			for i = 1; i <= (*m); i++ {
				work.Set(i-1, j-1, b.Get(i-1, (*n)-(*l)+j-1))
			}
		}
		err = goblas.Dtrmm(Right, Upper, NoTrans, NonUnit, *m, *l, one, v.Off(np-1, 0), *ldv, work, *ldwork)
		err = goblas.Dgemm(NoTrans, NoTrans, *m, *l, (*n)-(*l), one, b, *ldb, v, *ldv, one, work, *ldwork)
		err = goblas.Dgemm(NoTrans, NoTrans, *m, (*k)-(*l), *n, one, b, *ldb, v.Off(0, kp-1), *ldv, zero, work.Off(0, kp-1), *ldwork)

		for j = 1; j <= (*k); j++ {
			for i = 1; i <= (*m); i++ {
				work.Set(i-1, j-1, work.Get(i-1, j-1)+a.Get(i-1, j-1))
			}
		}

		err = goblas.Dtrmm(Right, Upper, mat.TransByte(trans), NonUnit, *m, *k, one, t, *ldt, work, *ldwork)

		for j = 1; j <= (*k); j++ {
			for i = 1; i <= (*m); i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		err = goblas.Dgemm(NoTrans, Trans, *m, (*n)-(*l), *k, -one, work, *ldwork, v, *ldv, one, b, *ldb)
		err = goblas.Dgemm(NoTrans, Trans, *m, *l, (*k)-(*l), -one, work.Off(0, kp-1), *ldwork, v.Off(np-1, kp-1), *ldv, one, b.Off(0, np-1), *ldb)
		err = goblas.Dtrmm(Right, Upper, Trans, NonUnit, *m, *l, one, v.Off(np-1, 0), *ldv, work, *ldwork)
		for j = 1; j <= (*l); j++ {
			for i = 1; i <= (*m); i++ {
				b.Set(i-1, (*n)-(*l)+j-1, b.Get(i-1, (*n)-(*l)+j-1)-work.Get(i-1, j-1))
			}
		}

	} else if column && backward && left {
		// ---------------------------------------------------------------------------
		//
		//        Let  W =  [ V ]    (M-by-K)
		//                  [ I ]    (K-by-K)
		//
		//        Form  H C  or  H**T C  where  C = [ B ]  (M-by-N)
		//                                          [ A ]  (K-by-N)
		//
		//        H = I - W T W**T          or  H**T = I - W T**T W**T
		//
		//        A = A -   T (A + V**T B)  or  A = A -   T**T (A + V**T B)
		//        B = B - V T (A + V**T B)  or  B = B - V T**T (A + V**T B)
		//
		// ---------------------------------------------------------------------------
		mp = minint((*l)+1, *m)
		kp = minint((*k)-(*l)+1, *k)

		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*l); i++ {
				work.Set((*k)-(*l)+i-1, j-1, b.Get(i-1, j-1))
			}
		}

		err = goblas.Dtrmm(Left, Lower, Trans, NonUnit, *l, *n, one, v.Off(0, kp-1), *ldv, work.Off(kp-1, 0), *ldwork)
		err = goblas.Dgemm(Trans, NoTrans, *l, *n, (*m)-(*l), one, v.Off(mp-1, kp-1), *ldv, b.Off(mp-1, 0), *ldb, one, work.Off(kp-1, 0), *ldwork)
		err = goblas.Dgemm(Trans, NoTrans, (*k)-(*l), *n, *m, one, v, *ldv, b, *ldb, zero, work, *ldwork)

		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*k); i++ {
				work.Set(i-1, j-1, work.Get(i-1, j-1)+a.Get(i-1, j-1))
			}
		}

		err = goblas.Dtrmm(Left, Lower, mat.TransByte(trans), NonUnit, *k, *n, one, t, *ldt, work, *ldwork)

		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*k); i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		err = goblas.Dgemm(NoTrans, NoTrans, (*m)-(*l), *n, *k, -one, v.Off(mp-1, 0), *ldv, work, *ldwork, one, b.Off(mp-1, 0), *ldb)
		err = goblas.Dgemm(NoTrans, NoTrans, *l, *n, (*k)-(*l), -one, v, *ldv, work, *ldwork, one, b, *ldb)
		err = goblas.Dtrmm(Left, Lower, NoTrans, NonUnit, *l, *n, one, v.Off(0, kp-1), *ldv, work.Off(kp-1, 0), *ldwork)
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*l); i++ {
				b.Set(i-1, j-1, b.Get(i-1, j-1)-work.Get((*k)-(*l)+i-1, j-1))
			}
		}

	} else if column && backward && right {
		// ---------------------------------------------------------------------------
		//
		//        Let  W =  [ V ]    (N-by-K)
		//                  [ I ]    (K-by-K)
		//
		//        Form  C H  or  C H**T  where  C = [ B A ] (B is M-by-N, A is M-by-K)
		//
		//        H = I - W T W**T          or  H**T = I - W T**T W**T
		//
		//        A = A - (A + B V) T      or  A = A - (A + B V) T**T
		//        B = B - (A + B V) T V**T  or  B = B - (A + B V) T**T V**T
		//
		// ---------------------------------------------------------------------------
		np = minint((*l)+1, *n)
		kp = minint((*k)-(*l)+1, *k)

		for j = 1; j <= (*l); j++ {
			for i = 1; i <= (*m); i++ {
				work.Set(i-1, (*k)-(*l)+j-1, b.Get(i-1, j-1))
			}
		}
		err = goblas.Dtrmm(Right, Lower, NoTrans, NonUnit, *m, *l, one, v.Off(0, kp-1), *ldv, work.Off(0, kp-1), *ldwork)
		err = goblas.Dgemm(NoTrans, NoTrans, *m, *l, (*n)-(*l), one, b.Off(0, np-1), *ldb, v.Off(np-1, kp-1), *ldv, one, work.Off(0, kp-1), *ldwork)
		err = goblas.Dgemm(NoTrans, NoTrans, *m, (*k)-(*l), *n, one, b, *ldb, v, *ldv, zero, work, *ldwork)

		for j = 1; j <= (*k); j++ {
			for i = 1; i <= (*m); i++ {
				work.Set(i-1, j-1, work.Get(i-1, j-1)+a.Get(i-1, j-1))
			}
		}

		err = goblas.Dtrmm(Right, Lower, mat.TransByte(trans), NonUnit, *m, *k, one, t, *ldt, work, *ldwork)

		for j = 1; j <= (*k); j++ {
			for i = 1; i <= (*m); i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		err = goblas.Dgemm(NoTrans, Trans, *m, (*n)-(*l), *k, -one, work, *ldwork, v.Off(np-1, 0), *ldv, one, b.Off(0, np-1), *ldb)
		err = goblas.Dgemm(NoTrans, Trans, *m, *l, (*k)-(*l), -one, work, *ldwork, v, *ldv, one, b, *ldb)
		err = goblas.Dtrmm(Right, Lower, Trans, NonUnit, *m, *l, one, v.Off(0, kp-1), *ldv, work.Off(0, kp-1), *ldwork)
		for j = 1; j <= (*l); j++ {
			for i = 1; i <= (*m); i++ {
				b.Set(i-1, j-1, b.Get(i-1, j-1)-work.Get(i-1, (*k)-(*l)+j-1))
			}
		}

	} else if row && forward && left {
		// ---------------------------------------------------------------------------
		//
		//        Let  W =  [ I V ] ( I is K-by-K, V is K-by-M )
		//
		//        Form  H C  or  H**T C  where  C = [ A ]  (K-by-N)
		//                                          [ B ]  (M-by-N)
		//
		//        H = I - W**T T W          or  H**T = I - W**T T**T W
		//
		//        A = A -     T (A + V B)  or  A = A -     T**T (A + V B)
		//        B = B - V**T T (A + V B)  or  B = B - V**T T**T (A + V B)
		//
		// ---------------------------------------------------------------------------
		mp = minint((*m)-(*l)+1, *m)
		kp = minint((*l)+1, *k)

		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*l); i++ {
				work.Set(i-1, j-1, b.Get((*m)-(*l)+i-1, j-1))
			}
		}
		err = goblas.Dtrmm(Left, Lower, NoTrans, NonUnit, *l, *n, one, v.Off(0, mp-1), *ldv, work, *ldb)
		err = goblas.Dgemm(NoTrans, NoTrans, *l, *n, (*m)-(*l), one, v, *ldv, b, *ldb, one, work, *ldwork)
		err = goblas.Dgemm(NoTrans, NoTrans, (*k)-(*l), *n, *m, one, v.Off(kp-1, 0), *ldv, b, *ldb, zero, work.Off(kp-1, 0), *ldwork)

		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*k); i++ {
				work.Set(i-1, j-1, work.Get(i-1, j-1)+a.Get(i-1, j-1))
			}
		}

		err = goblas.Dtrmm(Left, Upper, mat.TransByte(trans), NonUnit, *k, *n, one, t, *ldt, work, *ldwork)

		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*k); i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		err = goblas.Dgemm(Trans, NoTrans, (*m)-(*l), *n, *k, -one, v, *ldv, work, *ldwork, one, b, *ldb)
		err = goblas.Dgemm(Trans, NoTrans, *l, *n, (*k)-(*l), -one, v.Off(kp-1, mp-1), *ldv, work.Off(kp-1, 0), *ldwork, one, b.Off(mp-1, 0), *ldb)
		err = goblas.Dtrmm(Left, Lower, Trans, NonUnit, *l, *n, one, v.Off(0, mp-1), *ldv, work, *ldwork)
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*l); i++ {
				b.Set((*m)-(*l)+i-1, j-1, b.Get((*m)-(*l)+i-1, j-1)-work.Get(i-1, j-1))
			}
		}

	} else if row && forward && right {
		// ---------------------------------------------------------------------------
		//
		//        Let  W =  [ I V ] ( I is K-by-K, V is K-by-N )
		//
		//        Form  C H  or  C H**T  where  C = [ A B ] (A is M-by-K, B is M-by-N)
		//
		//        H = I - W**T T W            or  H**T = I - W**T T**T W
		//
		//        A = A - (A + B V**T) T      or  A = A - (A + B V**T) T**T
		//        B = B - (A + B V**T) T V    or  B = B - (A + B V**T) T**T V
		//
		// ---------------------------------------------------------------------------
		np = minint((*n)-(*l)+1, *n)
		kp = minint((*l)+1, *k)

		for j = 1; j <= (*l); j++ {
			for i = 1; i <= (*m); i++ {
				work.Set(i-1, j-1, b.Get(i-1, (*n)-(*l)+j-1))
			}
		}
		err = goblas.Dtrmm(Right, Lower, Trans, NonUnit, *m, *l, one, v.Off(0, np-1), *ldv, work, *ldwork)
		err = goblas.Dgemm(NoTrans, Trans, *m, *l, (*n)-(*l), one, b, *ldb, v, *ldv, one, work, *ldwork)
		err = goblas.Dgemm(NoTrans, Trans, *m, (*k)-(*l), *n, one, b, *ldb, v.Off(kp-1, 0), *ldv, zero, work.Off(0, kp-1), *ldwork)

		for j = 1; j <= (*k); j++ {
			for i = 1; i <= (*m); i++ {
				work.Set(i-1, j-1, work.Get(i-1, j-1)+a.Get(i-1, j-1))
			}
		}

		err = goblas.Dtrmm(Right, Upper, mat.TransByte(trans), NonUnit, *m, *k, one, t, *ldt, work, *ldwork)

		for j = 1; j <= (*k); j++ {
			for i = 1; i <= (*m); i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		err = goblas.Dgemm(NoTrans, NoTrans, *m, (*n)-(*l), *k, -one, work, *ldwork, v, *ldv, one, b, *ldb)
		err = goblas.Dgemm(NoTrans, NoTrans, *m, *l, (*k)-(*l), -one, work.Off(0, kp-1), *ldwork, v.Off(kp-1, np-1), *ldv, one, b.Off(0, np-1), *ldb)
		err = goblas.Dtrmm(Right, Lower, NoTrans, NonUnit, *m, *l, one, v.Off(0, np-1), *ldv, work, *ldwork)
		for j = 1; j <= (*l); j++ {
			for i = 1; i <= (*m); i++ {
				b.Set(i-1, (*n)-(*l)+j-1, b.Get(i-1, (*n)-(*l)+j-1)-work.Get(i-1, j-1))
			}
		}

	} else if row && backward && left {
		// ---------------------------------------------------------------------------
		//
		//        Let  W =  [ V I ] ( I is K-by-K, V is K-by-M )
		//
		//        Form  H C  or  H**T C  where  C = [ B ]  (M-by-N)
		//                                          [ A ]  (K-by-N)
		//
		//        H = I - W**T T W          or  H**T = I - W**T T**T W
		//
		//        A = A -     T (A + V B)  or  A = A -     T**T (A + V B)
		//        B = B - V**T T (A + V B)  or  B = B - V**T T**T (A + V B)
		//
		// ---------------------------------------------------------------------------
		mp = minint((*l)+1, *m)
		kp = minint((*k)-(*l)+1, *k)

		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*l); i++ {
				work.Set((*k)-(*l)+i-1, j-1, b.Get(i-1, j-1))
			}
		}
		err = goblas.Dtrmm(Left, Upper, NoTrans, NonUnit, *l, *n, one, v.Off(kp-1, 0), *ldv, work.Off(kp-1, 0), *ldwork)
		err = goblas.Dgemm(NoTrans, NoTrans, *l, *n, (*m)-(*l), one, v.Off(kp-1, mp-1), *ldv, b.Off(mp-1, 0), *ldb, one, work.Off(kp-1, 0), *ldwork)
		err = goblas.Dgemm(NoTrans, NoTrans, (*k)-(*l), *n, *m, one, v, *ldv, b, *ldb, zero, work, *ldwork)

		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*k); i++ {
				work.Set(i-1, j-1, work.Get(i-1, j-1)+a.Get(i-1, j-1))
			}
		}

		err = goblas.Dtrmm(Left, Lower, mat.TransByte(trans), NonUnit, *k, *n, one, t, *ldt, work, *ldwork)

		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*k); i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		err = goblas.Dgemm(Trans, NoTrans, (*m)-(*l), *n, *k, -one, v.Off(0, mp-1), *ldv, work, *ldwork, one, b.Off(mp-1, 0), *ldb)
		err = goblas.Dgemm(Trans, NoTrans, *l, *n, (*k)-(*l), -one, v, *ldv, work, *ldwork, one, b, *ldb)
		err = goblas.Dtrmm(Left, Upper, Trans, NonUnit, *l, *n, one, v.Off(kp-1, 0), *ldv, work.Off(kp-1, 0), *ldwork)
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*l); i++ {
				b.Set(i-1, j-1, b.Get(i-1, j-1)-work.Get((*k)-(*l)+i-1, j-1))
			}
		}

	} else if row && backward && right {
		// ---------------------------------------------------------------------------
		//
		//        Let  W =  [ V I ] ( I is K-by-K, V is K-by-N )
		//
		//        Form  C H  or  C H**T  where  C = [ B A ] (A is M-by-K, B is M-by-N)
		//
		//        H = I - W**T T W            or  H**T = I - W**T T**T W
		//
		//        A = A - (A + B V**T) T      or  A = A - (A + B V**T) T**T
		//        B = B - (A + B V**T) T V    or  B = B - (A + B V**T) T**T V
		//
		// ---------------------------------------------------------------------------
		np = minint((*l)+1, *n)
		kp = minint((*k)-(*l)+1, *k)

		for j = 1; j <= (*l); j++ {
			for i = 1; i <= (*m); i++ {
				work.Set(i-1, (*k)-(*l)+j-1, b.Get(i-1, j-1))
			}
		}
		err = goblas.Dtrmm(Right, Upper, Trans, NonUnit, *m, *l, one, v.Off(kp-1, 0), *ldv, work.Off(0, kp-1), *ldwork)
		err = goblas.Dgemm(NoTrans, Trans, *m, *l, (*n)-(*l), one, b.Off(0, np-1), *ldb, v.Off(kp-1, np-1), *ldv, one, work.Off(0, kp-1), *ldwork)
		err = goblas.Dgemm(NoTrans, Trans, *m, (*k)-(*l), *n, one, b, *ldb, v, *ldv, zero, work, *ldwork)

		for j = 1; j <= (*k); j++ {
			for i = 1; i <= (*m); i++ {
				work.Set(i-1, j-1, work.Get(i-1, j-1)+a.Get(i-1, j-1))
			}
		}

		err = goblas.Dtrmm(Right, Lower, mat.TransByte(trans), NonUnit, *m, *k, one, t, *ldt, work, *ldwork)

		for j = 1; j <= (*k); j++ {
			for i = 1; i <= (*m); i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		err = goblas.Dgemm(NoTrans, NoTrans, *m, (*n)-(*l), *k, -one, work, *ldwork, v.Off(0, np-1), *ldv, one, b.Off(0, np-1), *ldb)
		err = goblas.Dgemm(NoTrans, NoTrans, *m, *l, (*k)-(*l), -one, work, *ldwork, v, *ldv, one, b, *ldb)
		err = goblas.Dtrmm(Right, Upper, NoTrans, NonUnit, *m, *l, one, v.Off(kp-1, 0), *ldv, work.Off(0, kp-1), *ldwork)
		for j = 1; j <= (*l); j++ {
			for i = 1; i <= (*m); i++ {
				b.Set(i-1, j-1, b.Get(i-1, j-1)-work.Get(i-1, (*k)-(*l)+j-1))
			}
		}

	}
}
